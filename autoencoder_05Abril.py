import os
import gc
import json
import glob
import torchc
import resource
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
import umap

# ==============================================================================
# 0. CONFIGURACIÓN Y UTILIDADES VECTORIZADAS
# ==============================================================================
def configure_system_resources(max_ram_gb=100, target_cores=14):
    limit_bytes = int(max_ram_gb * 1024**3)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
    except Exception as e: print(f"[!] RAM Limit error: {e}")

    available_cores = multiprocessing.cpu_count() if 'multiprocessing' in globals() else 14
    use_cores = min(target_cores, available_cores)
    torch.set_num_threads(use_cores)
    os.environ["OMP_NUM_THREADS"] = str(use_cores)
    return use_cores

def fast_parse_vectorized(df_col, max_p):
    df_col = df_col.str.replace('"', '', regex=False)
    expanded = df_col.str.split(';', expand=True).iloc[:, :max_p].astype(np.float32).fillna(0.0)
    if expanded.shape[1] < max_p:
        padding = np.zeros((len(expanded), max_p - expanded.shape[1]), dtype=np.float32)
        return np.hstack([expanded.values, padding])
    return expanded.values

def calculate_jet_axis_vectorized(pt, eta, phi):
    sum_pt = np.sum(pt, axis=1, keepdims=True) + 1e-8
    w_eta = np.sum(pt * eta, axis=1, keepdims=True) / sum_pt
    w_x = np.sum(pt * np.cos(phi), axis=1, keepdims=True)
    w_y = np.sum(pt * np.sin(phi), axis=1, keepdims=True)
    w_phi = np.arctan2(w_y, w_x)
    return w_eta, w_phi

# ==============================================================================
# 1. PREPROCESADOR (OPTIMIZADO)
# ==============================================================================
class JetPreprocessorCSV:
    def __init__(self, max_particles=30):
        self.max_particles = max_particles
        self.mean, self.std = None, None

    def fit(self, file_pattern):
        print("[*] Calculando estadísticas vectorizadas...")
        files = glob.glob(file_pattern)
        if not files: return
        
        df = pd.read_csv(files[0], usecols=['PF_Pt', 'PF_Eta', 'PF_Phi'], nrows=100000)
        pt = fast_parse_vectorized(df['PF_Pt'], self.max_particles)
        eta = fast_parse_vectorized(df['PF_Eta'], self.max_particles)
        phi = fast_parse_vectorized(df['PF_Phi'], self.max_particles)
        
        j_eta, j_phi = calculate_jet_axis_vectorized(pt, eta, phi)
        logpt = np.log1p(pt)
        deta = (eta - j_eta) * (pt > 0)
        dphi = ((phi - j_phi + np.pi) % (2 * np.pi) - np.pi) * (pt > 0)
        
        mask = pt > 0
        self.mean = np.array([logpt[mask].mean(), deta[mask].mean(), dphi[mask].mean()], dtype=np.float32)
        self.std = np.array([logpt[mask].std(), deta[mask].std(), dphi[mask].std()], dtype=np.float32)
        print(f"   Mean: {self.mean} | Std: {self.std}")

    def save(self, path="scaler_csv.json"):
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f)

    def load(self, path="scaler_csv.json"):
        with open(path, "r") as f:
            d = json.load(f)
            self.mean, self.std = np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32)

# ==============================================================================
# 2. DATASET CON CACHÉ BINARIO
# ==============================================================================
class JetDatasetOptimized(Dataset):
    def __init__(self, file_path, preprocessor, max_particles=30, force_reprocess=False, max_rows=None):
        self.cache_path = file_path.replace(".csv", "_cache.pt")
        
        if force_reprocess and os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            print(f"[!] Caché antiguo eliminado: {os.path.basename(self.cache_path)}")
        
        if os.path.exists(self.cache_path) and not force_reprocess:
            print(f"[>] Cargando caché actualizado: {os.path.basename(self.cache_path)}")
            self.data, self.masks, self.physics = torch.load(self.cache_path)
            if max_rows is not None and len(self.data) > max_rows:
                self.data = self.data[:max_rows]
                self.masks = self.masks[:max_rows]
                self.physics = self.physics[:max_rows]
        else:
            print(f"[>] Procesando vectorialmente y extrayendo física: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path, nrows=max_rows) 
            
            pt = fast_parse_vectorized(df['PF_Pt'], max_particles)
            eta = fast_parse_vectorized(df['PF_Eta'], max_particles)
            phi = fast_parse_vectorized(df['PF_Phi'], max_particles)
            
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            pz = pt * np.sinh(eta)
            e = pt * np.cosh(eta)
            
            jet_px, jet_py = np.sum(px, axis=1), np.sum(py, axis=1) 
            jet_pz, jet_e = np.sum(pz, axis=1), np.sum(e, axis=1)
            
            jet_mass2 = jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2
            jet_mass = np.sqrt(np.maximum(0, jet_mass2))
            jet_pt = np.sqrt(jet_px**2 + jet_py**2)

            tau1 = df['Tau1'].values.astype(np.float32) + 1e-8
            tau2 = df['Tau2'].values.astype(np.float32)
            tau21 = tau2 / tau1
            SoftMass = df['SDMass'].values.astype(np.float32)
            
            n_constituents = np.sum(pt > 0, axis=1)
            # Orden físico: 0:Mass, 1:Pt, 2:N, 3:Tau21, 4:SoftMass
            physics_vars = np.stack([jet_mass, jet_pt, n_constituents, tau21, SoftMass], axis=1).astype(np.float32)
            
            j_eta, j_phi = calculate_jet_axis_vectorized(pt, eta, phi)
            logpt = np.log1p(pt)
            deta = (eta - j_eta)
            dphi = (phi - j_phi + np.pi) % (2 * np.pi) - np.pi
            
            features = np.stack([logpt, deta, dphi], axis=2).astype(np.float32)
            features = (features - preprocessor.mean) / (preprocessor.std + 1e-8)
            masks = (pt > 0).astype(np.float32)
            features = features * np.expand_dims(masks, axis=2) 
            
            self.data = torch.from_numpy(features)
            self.masks = torch.from_numpy(masks)
            self.physics = torch.from_numpy(physics_vars)
            
            torch.save((self.data, self.masks, self.physics), self.cache_path)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.masks[idx], self.physics[idx]

# ==============================================================================
# 3. MODELO, DISTANCIA Y PENALIZACIÓN DISCO
# ==============================================================================

# MODIFICACIÓN 1: Permite calcular la pérdida por jet individualmente
def chamfer_distance_fast(p1, p2, mask1, mask2, reduction='mean'):
    """Usa torch.cdist para máxima velocidad en CPU/GPU."""
    dist = torch.cdist(p1, p2, p=2)**2
    
    m1, m2 = mask1.unsqueeze(2), mask2.unsqueeze(1)
    dist = dist + (1 - m1) * 1e8 + (1 - m2) * 1e8

    loss_12 = torch.min(dist, dim=2)[0] * mask1 # [B, N]
    loss_21 = torch.min(dist, dim=1)[0] * mask2 # [B, N]

    # Pérdida total por cada jet
    per_jet_loss = loss_12.sum(dim=1) + loss_21.sum(dim=1)
    
    # Normalizar por el número de partículas válidas en ese jet
    n_particles = mask1.sum(dim=1) + 1e-8
    per_jet_loss = per_jet_loss / n_particles

    if reduction == 'none':
        return per_jet_loss # Devuelve vector [Batch_size] para DisCo
    return per_jet_loss.mean() # Devuelve escalar promedio

# MODIFICACIÓN 2: Implementación Matemática de Distance Correlation (DisCo)
def calc_distance_correlation(x, y):
    """
    Calcula DisCo entre la pérdida de reconstrucción y la variable cinemática.
    x: Tensor de pérdida (Loss) de tamaño [B]
    y: Tensor de masa (Mass) de tamaño [B]
    """
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    
    # Cálculo de matrices de distancia por pares
    A = torch.abs(x - x.t())
    B = torch.abs(y - y.t())
    
    # Doble centrado para calcular la covarianza de distancia
    A_mean_row = A.mean(dim=0, keepdim=True)
    A_mean_col = A.mean(dim=1, keepdim=True)
    A_mean_all = A.mean()
    A_c = A - A_mean_row - A_mean_col + A_mean_all
    
    B_mean_row = B.mean(dim=0, keepdim=True)
    B_mean_col = B.mean(dim=1, keepdim=True)
    B_mean_all = B.mean()
    B_c = B - B_mean_row - B_mean_col + B_mean_all
    
    dcov2_xy = (A_c * B_c).mean()
    dcov2_xx = (A_c * A_c).mean()
    dcov2_yy = (B_c * B_c).mean()
    
    # Correlación de distancia (al cuadrado por estabilidad numérica)
    dcor2 = dcov2_xy / (torch.sqrt(dcov2_xx * dcov2_yy) + 1e-8)
    return dcor2

class EdgeConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=16):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        b, c, n = x.shape
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        dist = -xx - inner - xx.transpose(2, 1)
        idx = dist.topk(k=self.k, dim=-1)[1]
        
        idx_base = torch.arange(0, b, device=x.device).view(-1, 1, 1) * n
        idx = (idx + idx_base).view(-1)
        x_flat = x.transpose(2, 1).contiguous().view(b * n, c)
        neighbors = x_flat[idx, :].view(b, n, self.k, c).permute(0, 3, 1, 2)
        x_cent = x.unsqueeze(3).repeat(1, 1, 1, self.k)
        out = self.conv(torch.cat([x_cent, neighbors - x_cent], dim=1)).mean(dim=3)
        return out + x if c == out.shape[1] else out

class ParticleNetAE(nn.Module):
    def __init__(self, input_dim=3, n_part=30, latent=16):
        super().__init__()
        self.n_part = n_part
        self.enc = nn.Sequential(EdgeConv(input_dim, 64), EdgeConv(64, 128), EdgeConv(128, 256))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_z = nn.Linear(256, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, n_part * input_dim)
        )
    def forward(self, x):
        z = self.fc_z(self.pool(self.enc(x.transpose(1, 2))).squeeze(-1))
        return self.dec(z).view(-1, self.n_part, 3)

    def encode(self, x):
        return self.fc_z(self.pool(self.enc(x.transpose(1, 2))).squeeze(-1))

def analyze_and_plot_umap(model, val_loader, device):
    print("[*] Extrayendo representaciones latentes para UMAP...")
    model.eval()
    all_z, all_phys = [], []
    
    with torch.no_grad():
        for x, _, phys in val_loader:
            x = x.to(device)
            if hasattr(model, '_orig_mod'):
                z = model._orig_mod.encode(x)
            else:
                z = model.encode(x)
            
            all_z.append(z.cpu().numpy())
            all_phys.append(phys.numpy())
            
    Z = np.concatenate(all_z, axis=0)
    phys_matrix = np.concatenate(all_phys, axis=0)
    
    print("[*] Calculando proyección UMAP (solo se calcula una vez)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(Z)
    
    variables = {
        0: {"name": "Masa Invariante", "unit": "GeV", "cmap": "viridis", "file": "umap_mass.png"},
        1: {"name": "Momento Transversal (pT)", "unit": "GeV", "cmap": "plasma", "file": "umap_pt.png"},
        2: {"name": "Multiplicidad (N const)", "unit": "", "cmap": "magma", "file": "umap_n.png"},
        3: {"name": "N-subjettiness (Tau21)", "unit": "", "cmap": "magma", "file": "umap_tau21.png"},
        4: {"name": "Softdrop Mass (MSoft)", "unit": "GeV", "cmap": "viridis", "file": "umap_Msoft.png"},
    }
    
    print("[*] Generando gráficas coloreadas...")
    for idx, config in variables.items():
        plt.figure(figsize=(10, 8))
        c_values = phys_matrix[:, idx]
        
        vmin, vmax = np.percentile(c_values, [5, 95]) 
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=c_values, 
                              cmap=config["cmap"], s=5, alpha=0.8, vmin=vmin, vmax=vmax)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label(f'{config["name"]} [{config["unit"]}]' if config["unit"] else config["name"])
        
        plt.title(f'Espacio Latente (Coloreado por {config["name"]})')
        plt.xlabel('UMAP Dim 1')
        plt.ylabel('UMAP Dim 2')
        plt.tight_layout()
        plt.savefig(config["file"], dpi=300)
        plt.show() # Descomentar si corres en Jupyter
        plt.close()
        print(f"  -> Guardado: {config['file']}")

# ==============================================================================
# 4. MAIN Y BUCLE DE ENTRENAMIENTO DISCO
# ==============================================================================
def main():
    configure_system_resources()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MAX_PARTICLES = 30
    BATCH_SIZE = 256
    TRAIN_PATTERN = "file_0.csv"
    
    prep = JetPreprocessorCSV(MAX_PARTICLES)
    if os.path.exists("scaler_csv.json"): prep.load()
    else: 
        prep.fit(TRAIN_PATTERN)
        prep.save()

    files = glob.glob(TRAIN_PATTERN)[:1]
    datasets = [JetDatasetOptimized(f, prep, MAX_PARTICLES, force_reprocess=True, max_rows=30000) for f in files]
    full_dataset = ConcatDataset(datasets)

    train_size = int(0.9 * len(full_dataset))
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)

    model = ParticleNetAE(n_part=MAX_PARTICLES).to(device)
    
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("[*] Modelo compilado con torch.compile")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = [], []
    train_disco_vals = [] # Para monitorear cómo baja la correlación
    
    num_epochs = 20 
    
    # MODIFICACIÓN 3: Hiperparámetro Lambda para controlar la fuerza de la descorrelación
    # Deberás ajustarlo empíricamente. Si es muy bajo, esculpe masa. Si es muy alto, reconstruye mal.
    LAMBDA_DISCO = 5.0 

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        running_train_loss = 0.0
        running_disco_val = 0.0

        # Ahora sí desempaquetamos la matriz física 'phys' en el bucle
        for x, m, phys in pbar:    
            x, m, phys = x.to(device), m.to(device), phys.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            recon = model(x)
            
            # 1. Pérdida de reconstrucción (obtenemos vector [Batch_size])
            per_jet_recon_loss = chamfer_distance_fast(x, recon, m, m, reduction='none')
            mean_recon_loss = per_jet_recon_loss.mean()
            
            # 2. Extraemos la Masa para DisCo (Índice 4 = SoftMass, Índice 0 = Masa Invariante)
            masses = phys[:, 4] 
            
            # 3. Calculamos la penalización DisCo
            disco_penalty = calc_distance_correlation(per_jet_recon_loss, masses)
            
            # 4. Pérdida Total
            total_loss = mean_recon_loss + (LAMBDA_DISCO * disco_penalty)
            
            total_loss.backward()
            optimizer.step()
            
            running_train_loss += mean_recon_loss.item()
            running_disco_val += disco_penalty.item()
            
            pbar.set_postfix(Recon=f"{mean_recon_loss.item():.4f}", DisCo=f"{disco_penalty.item():.4f}")
            
        avg_train_loss = running_train_loss / len(train_loader)
        avg_disco_val = running_disco_val / len(train_loader)
        train_losses.append(avg_train_loss)
        train_disco_vals.append(avg_disco_val)

        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for x, m, _ in val_loader: 
                x, m = x.to(device), m.to(device)
                recon = model(x)
                # En validación solo nos interesa monitorear la reconstrucción pura
                loss = chamfer_distance_fast(x, recon, m, m, reduction='mean')
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"[*] Fin Epoch {epoch+1} | Val Recon Loss: {avg_val_loss:.4f} | Train DisCo: {avg_disco_val:.4f}")

    print("[*] Generando gráfica de entrenamiento...")
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Recon')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Recon')
    plt.title('Reconstruction Loss (Chamfer)')
    plt.xlabel('Época')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_disco_vals, color='orange', label='DisCo Penalty')
    plt.title('Evolución de Correlación (DisCo)')
    plt.xlabel('Época')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('grafica_entrenamiento_disco.png', dpi=300)
    plt.show()
    plt.close()

    analyze_and_plot_umap(model, val_loader, device)

if __name__ == "__main__":
    main()
