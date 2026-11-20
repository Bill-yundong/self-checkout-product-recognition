"""Verify the local environment and report missing optional dependencies."""
def check_module(module_name: str, required: bool) -> None:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {module_name} {version}")
    except ImportError:
        label = "required" if required else "optional"
        print(f"[MISSING] {module_name} ({label})")


def main() -> None:
    for module in ["torch", "torchvision", "cv2", "PIL", "matplotlib", "reportlab"]:
        check_module(module, required=True)
    for module in ["ultralytics", "yaml"]:
        check_module(module, required=False)


if __name__ == "__main__":
    main()
