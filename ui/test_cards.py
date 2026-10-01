import traceback

try:
    import ui.cards
    print("SUCCESS")
except Exception:
    traceback.print_exc()