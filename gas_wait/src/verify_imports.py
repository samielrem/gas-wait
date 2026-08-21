"""Verify that core data science packages can be imported."""


def main() -> None:
    packages = [
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "requests",
    ]

    for name in packages:
        __import__(name)
        print(f"OK: {name}")

    print("All required packages imported successfully.")


if __name__ == "__main__":
    main()
