# debug_astm.py
import math

mean_intercept_mm = 0.017112
log_val = math.log10(mean_intercept_mm)
print(f"log10({mean_intercept_mm}) = {log_val}")

G_correct  = (-6.6457 * log_val) - 3.298
G_wrong    = (6.643856 * log_val) - 3.288

print(f"G with negative coefficient = {G_correct:.2f}")
print(f"G with positive coefficient = {G_wrong:.2f}")