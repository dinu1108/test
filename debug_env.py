import sys
import os
print(f"Exec: {sys.executable}")
print(f"Bina: {os.path.dirname(sys.executable)}")
try:
    import chromadb
    print("CHROMA: OK")
except ImportError as e:
    print(f"CHROMA: {e}")
try:
    import langgraph
    print("LANGGRAPH: OK")
except ImportError as e:
    print(f"LANGGRAPH: {e}")
