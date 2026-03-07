"""Run this to find exactly which predictor.py the engine is using."""
import sys, os

# Show Python being used
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working dir: {os.getcwd()}")
print()

# Show where predictor is being imported from
import predictor
print(f"predictor.py loaded from: {predictor.__file__}")
print()

# Check the actual draw blend value in the loaded module
import inspect
src = inspect.getsource(predictor.Predictor.predict)
if "0.20 if market == " in src:
    print("CORRECT: draw blend=0.20 found in loaded predictor")
elif "effective_blend = 0.30" in src and "0.20" not in src:
    print("OLD CODE: draw blend still 0.30 — wrong file being loaded")
else:
    print("UNKNOWN: check predictor manually")
print()

# Check the raw edge floor value
if "0.09" in src:
    print("CORRECT: raw floor 0.09 found")
elif "0.07" in src:
    print("OLD CODE: raw floor still 0.07")
else:
    print("UNKNOWN: check raw floor manually")
print()

# Quick prediction test
p = predictor.Predictor()
row = {
    "home": "AEK Athens", "away": "Larissa FC",
    "league": "Super League Greece",
    "regime": "neutral", "drs": 0.12, "motivation_delta": 0,
    "odds_1": 1.35, "odds_X": 6.50, "odds_2": 8.00,
}
for pr in p.predict(row):
    if pr.market == "draw":
        print(f"AEK draw: p_true={pr.p_true:.3f}  has_edge={pr.has_edge}")
        if pr.p_true < 0.20:
            print("-> NEW predictor active (correct)")
        else:
            print("-> OLD predictor active (wrong file!)")

# List all predictor.py files on disk that Python can find
print()
print("All predictor.py files Python can see:")
for path in sys.path:
    candidate = os.path.join(path, "predictor.py")
    if os.path.exists(candidate):
        print(f"  {candidate}")
