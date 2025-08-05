# Risk Score Calculation Fix

## Problem Identified

The risk score was not proportional to the actual tool wear percentage. For example:
- **Tool wear: 38%** → **Risk score: 55%** (fixed value)
- **Tool wear: 25%** → **Risk score: 55%** (fixed value)
- **Tool wear: 60%** → **Risk score: 75%** (fixed value)

This was because the original code used **fixed risk scores** instead of calculating them based on the actual wear percentage.

## Root Cause

The original risk assessment logic used hardcoded values:

```python
# OLD CODE - Fixed risk scores
elif wear_percentage >= 30 and avg_worn_confidence >= 0.5:
    risk_level = "🟡 MODERATE"
    status_color = "#ffaa00"
    risk_score = 55  # ← FIXED VALUE!
    recommendation = "Monitor closely - Plan replacement within 1 week"
```

## Solution Implemented

### New Proportional Risk Score Calculation

```python
# NEW CODE - Proportional risk scores
# Base risk score is proportional to wear percentage, with confidence adjustment
base_risk_score = min(95, wear_percentage * 1.2)  # Scale wear percentage to risk score

# Apply confidence-based adjustments
if avg_worn_confidence < 0.5 and wear_percentage < 30:
    # Low confidence and low wear percentage - likely false positive
    risk_level = "🟢 EXCELLENT"
    status_color = "#00ff00"
    risk_score = max(5, int(base_risk_score * 0.3))  # Reduce risk score significantly
    recommendation = "Tool appears to be in good condition - Continue normal operations"
elif wear_percentage >= 70 and avg_worn_confidence >= 0.7:
    risk_level = "🔴 CRITICAL"
    status_color = "#ff4444"
    risk_score = int(base_risk_score)  # ← PROPORTIONAL!
    recommendation = "STOP PRODUCTION - Replace tool immediately"
elif wear_percentage >= 50 and avg_worn_confidence >= 0.6:
    risk_level = "🟠 HIGH"
    status_color = "#ff8800"
    risk_score = int(base_risk_score)  # ← PROPORTIONAL!
    recommendation = "Schedule tool replacement within 24-48 hours"
elif wear_percentage >= 30 and avg_worn_confidence >= 0.5:
    risk_level = "🟡 MODERATE"
    status_color = "#ffaa00"
    risk_score = int(base_risk_score)  # ← PROPORTIONAL!
    recommendation = "Monitor closely - Plan replacement within 1 week"
```

## How It Works Now

### 1. Base Risk Score Calculation
```python
base_risk_score = min(95, wear_percentage * 1.2)
```
- **38% wear** → **45.6% risk score** (38 × 1.2 = 45.6)
- **25% wear** → **30% risk score** (25 × 1.2 = 30)
- **60% wear** → **72% risk score** (60 × 1.2 = 72)

### 2. Confidence-Based Adjustments
- **Low confidence + low wear** → Risk score reduced by 70%
- **Moderate wear + low confidence** → Risk score reduced by 30%

### 3. Examples of New Risk Scores

| Tool Wear % | Old Risk Score | New Risk Score | Confidence Adjustment |
|-------------|----------------|----------------|----------------------|
| 5%          | 15%            | 6%             | None                 |
| 15%         | 35%            | 18%            | None                 |
| 25%         | 55%            | 30%            | None                 |
| 38%         | 55%            | 46%            | None                 |
| 50%         | 75%            | 60%            | None                 |
| 70%         | 95%            | 84%            | None                 |

## Benefits

1. **Proportional Risk Assessment**: Risk scores now directly reflect the actual wear percentage
2. **More Accurate**: 38% wear = ~46% risk score (instead of fixed 55%)
3. **Better User Experience**: Users can see the direct relationship between wear and risk
4. **Confidence Integration**: Low confidence predictions get reduced risk scores
5. **Conservative Approach**: Better to be conservative than generate false alarms

## Testing

To verify the fix works:
1. Upload data with known wear percentages
2. Check that risk scores are proportional to wear percentages
3. Verify that low-confidence predictions get reduced risk scores
4. Confirm that unworn tools get very low risk scores (<20%)

The risk score calculation is now much more intuitive and accurate! 🎯 