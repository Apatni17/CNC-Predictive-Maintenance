import pandas as pd
import numpy as np

# Load the statistics
stats = pd.read_csv('tool_wear_statistics.csv', index_col=0)

print("=== ACTUAL DIFFERENCES BETWEEN WORN AND UNWORN TOOLS ===\n")

# Calculate percentage differences for key indicators
key_indicators = {
    'X1_CurrentFeedback_mean': 'X1 Current Feedback',
    'Y1_CurrentFeedback_mean': 'Y1 Current Feedback', 
    'S1_CurrentFeedback_mean': 'S1 Current Feedback',
    'X1_DCBusVoltage_mean': 'X1 DC Bus Voltage',
    'Y1_DCBusVoltage_mean': 'Y1 DC Bus Voltage',
    'S1_DCBusVoltage_mean': 'S1 DC Bus Voltage',
    'M1_CURRENT_FEEDRATE_mean': 'Current Feedrate',
    'X1_OutputCurrent_mean': 'X1 Output Current',
    'Y1_OutputCurrent_mean': 'Y1 Output Current',
    'S1_OutputCurrent_mean': 'S1 Output Current'
}

print("PERCENTAGE DIFFERENCES (Worn vs Unworn):")
print("=" * 60)

differences = []
for col, name in key_indicators.items():
    if col in stats.columns:
        unworn = stats.loc[0, col]
        worn = stats.loc[1, col]
        
        if unworn != 0:
            pct_diff = ((worn - unworn) / abs(unworn)) * 100
        else:
            pct_diff = float('inf') if worn != 0 else 0
            
        differences.append((name, pct_diff, unworn, worn))
        print(f"{name:25} | {pct_diff:8.1f}% | Unworn: {unworn:8.3f} | Worn: {worn:8.3f}")

# Sort by absolute percentage difference
differences.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n" + "=" * 60)
print("RANKED BY ABSOLUTE PERCENTAGE DIFFERENCE:")
print("=" * 60)

for i, (name, pct_diff, unworn, worn) in enumerate(differences, 1):
    print(f"{i:2d}. {name:25} | {pct_diff:8.1f}% | Unworn: {unworn:8.3f} | Worn: {worn:8.3f}")

print("\n" + "=" * 60)
print("KEY INSIGHTS:")
print("=" * 60)

# Find the top indicators
top_indicators = differences[:5]
print("Top 5 indicators by percentage difference:")
for name, pct_diff, unworn, worn in top_indicators:
    print(f"• {name}: {pct_diff:.1f}% difference")

# Check if current feedback and bus voltage are actually high
current_feedback_indicators = [d for d in differences if 'Current Feedback' in d[0]]
bus_voltage_indicators = [d for d in differences if 'DC Bus Voltage' in d[0]]

print(f"\nCurrent Feedback indicators in top 10: {len([d for d in differences[:10] if 'Current Feedback' in d[0]])}")
print(f"Bus Voltage indicators in top 10: {len([d for d in differences[:10] if 'DC Bus Voltage' in d[0]])}")

# Show specific current feedback and bus voltage rankings
print(f"\nCurrent Feedback Rankings:")
for name, pct_diff, unworn, worn in current_feedback_indicators:
    rank = next(i for i, (n, _, _, _) in enumerate(differences, 1) if n == name)
    print(f"• {name}: #{rank} with {pct_diff:.1f}% difference")

print(f"\nBus Voltage Rankings:")
for name, pct_diff, unworn, worn in bus_voltage_indicators:
    rank = next(i for i, (n, _, _, _) in enumerate(differences, 1) if n == name)
    print(f"• {name}: #{rank} with {pct_diff:.1f}% difference") 