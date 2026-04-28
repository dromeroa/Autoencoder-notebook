import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_mass_sculpting(df, mass_col='mass', score_col='score', 
                        working_points=[100, 10, 5, 1], bins=50, mass_range=(40, 200),
                        save_name='mass_sculpting.png'):
    """
    Evalúa el mass sculpting trazando la masa invariante del fondo a diferentes 
    cortes (percentiles) del score del clasificador y guarda el gráfico como PNG.
    """
    plt.figure(figsize=(10, 7))
    
    # Iteramos sobre cada punto de trabajo (porcentaje de retención)
    for wp in working_points:
        if wp == 100:
            cut_mask = df[score_col] > -np.inf
            label = 'Sin corte (100% del fondo)'
        else:
            threshold = np.percentile(df[score_col], 100 - wp)
            cut_mask = df[score_col] > threshold
            label = f'Corte top {wp}% (Score > {threshold:.3f})'
            
        # Filtramos las masas que pasan el corte
        masses_passing_cut = df.loc[cut_mask, mass_col]
        
        # Graficamos con density=True para comparar formas
        plt.hist(masses_passing_cut, bins=bins, range=mass_range, 
                 density=True, histtype='step', linewidth=2, label=label)

    # Formato gráfico tipo HEP
    plt.title('Evaluación de Mass Sculpting en el Fondo', fontsize=14, pad=15)
    plt.xlabel(f'Masa Invariante del FatJet [{mass_col}]', fontsize=12)
    plt.ylabel('Densidad de Eventos Normalizada (a.u.)', fontsize=12)
    plt.xlim(mass_range)
    plt.legend(loc='upper right', frameon=False, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # --- BLOQUE PARA GUARDAR LA IMAGEN (Debe ir antes de plt.show) ---
    if save_name:
        # dpi=300 asegura calidad de publicación. bbox_inches elimina márgenes blancos extra.
        plt.savefig(save_name, dpi=300, bbox_inches='tight', transparent=False)
        print(f"✅ Gráfico guardado exitosamente como: {save_name}")
    # -----------------------------------------------------------------
    
    plt.show()

# ==========================================
# EJEMPLO DE USO CON TUS DATOS
# ==========================================
if __name__ == "__main__":
    # Suponiendo que ya tienes tu DataFrame df_bkg cargado con las columnas correctas
    # Simulación rápida para que el script no falle si lo copias y pegas:
    np.random.seed(42)
    df_bkg = pd.DataFrame({
        'FatJet_mass': np.random.exponential(scale=30, size=100000) + 40,
        'NN_score': np.random.beta(a=2, b=5, size=100000)
    })
    
    print("Generando y guardando el gráfico...")
    
    # Llamamos a la función. 
    # Al no pasarle 'save_name', usará el valor por defecto: 'mass_sculpting.png'
    plot_mass_sculpting(df_bkg, mass_col='FatJet_mass', score_col='NN_score')
    
    # Si quisieras guardarlo con otro nombre específico, lo harías así:
    # plot_mass_sculpting(df_bkg, mass_col='FatJet_mass', score_col='NN_score', save_name='mi_analisis_v2.png')
