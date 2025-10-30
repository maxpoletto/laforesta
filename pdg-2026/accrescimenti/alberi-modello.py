import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

data = [
    # Genere, diametro a cm 150 (cm), altezza (m)
    ('Abete', 40, 27),
    ('Abete', 39, 27),
    ('Abete', 37, 27),
    ('Abete', 38, 26),
    ('Abete', 41, 26),
    ('Abete', 37, 23),
    ('Abete', 32, 28),
    ('Abete', 31, 23),
    ('Abete', 22, 18),
    ('Abete', 37, 28),
    ('Abete', 5, 26),
    ('Abete', 35, 26),
    ('Abete', 30, 20),
    ('Abete', 32, 15),
    ('Abete', 24, 16),
    ('Douglasia', 52, 27),
    ('Douglasia', 33, 31),
    ('Douglasia', 28, 21),
    ('Douglasia', 32, 28),
    ('Douglasia', 29, 25),
    ('Faggio', 27, 24),
    ('Faggio', 27, 25),
    ('Faggio', 32, 26),
    ('Faggio', 28, 27),
    ('Faggio', 29, 23),
    ('Faggio', 27, 21),
    ('Faggio', 33, 23),
    ('Faggio', 34, 24),
    ('Faggio', 30, 18),
    ('Faggio', 31, 23),
    ('Faggio', 27, 20),
    ('Faggio', 25, 15),
    ('Faggio', 28, 20),
    ('Faggio', 27, 24),
    ('Faggio', 25, 21),
    ('Faggio', 25, 18),
    ('Faggio', 30, 21),
    ('Faggio', 29, 27),
    ('Faggio', 36, 25),
    ('Faggio', 33, 25),
    ('Faggio', 31, 21),
    ('Faggio', 26, 16),
    ('Faggio', 27, 18),
    ('Faggio', 27, 18),
    ('Faggio', 27, 18),
    ('Ontano', 20, 15),
    ('Ontano', 23, 15),
    ('Ontano', 25, 14),
    ('Pino', 31, 20),
    ('Pino', 31, 23),
    ('Pino', 27, 20),
    ('Pino', 30, 23),
    ('Pino Marittimo', 28, 21),
    ('Pino Marittimo', 31, 21),
    ('Pino Nero', 43, 28),
    ('Pino Nero', 46, 28),
    ('Pino Nero', 34, 21),
    ('Pino Nero', 34, 21),
    ('Pino Nero', 34, 31),
    ('Pino Nero', 35, 25),
    ('Pino Nero', 38, 24),
    ('Pino Nero', 35, 25),
    ('Pino Nero', 32, 19),
]

df = pd.DataFrame(data, columns=['Specie', 'Diametro_cm', 'Altezza_m'])

plt.figure(figsize=(12, 7))

species = df['Specie'].unique()

# Colors for consistent plotting
colors = plt.cm.tab10(np.linspace(0, 1, len(species)))

def log_func(x, a, b):
    return a * np.log(x) + b

for i, sp in enumerate(species):
    species_data = df[df['Specie'] == sp]
    x = species_data['Diametro_cm'].values
    y = species_data['Altezza_m'].values
    
    plt.scatter(x, y, label=sp, alpha=0.7, s=50, color=colors[i])
    
    if len(x) >= 3:
        try:
            # Fit the curve
            popt, _ = curve_fit(log_func, x, y)
            a, b = popt
            
            # Calculate R²
            y_pred = log_func(x, a, b)
            r2 = r2_score(y, y_pred)
            
            # Generate smooth curve for plotting
            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = log_func(x_smooth, a, b)
            
            # Plot the fitted curve
            plt.plot(x_smooth, y_smooth, '--', color=colors[i], 
                    alpha=0.6, linewidth=2,
                    label=f'{sp} fit (R²={r2:.3f})')
            
            print(f"\n{sp}:")
            print(f"  n = {len(x)} points")
            print(f"  y = {a:.3f}·ln(x) + {b:.3f}")
            print(f"  R² = {r2:.3f}")
            
        except Exception as e:
            print(f"\n{sp}: Could not fit curve - {e}")
    else:
        print(f"\n{sp}: Too few points ({len(x)}) for fitting")

plt.xlabel('Diametro a cm 150 (cm)', fontsize=11)
plt.ylabel('Altezza (m)', fontsize=11)
plt.title('Altezza vs Diametro per Specie (con fit logaritmico)', fontsize=13)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('alberi-modello.png', dpi=150, bbox_inches='tight')
print("Plot salvato in alberi-modello.png")
