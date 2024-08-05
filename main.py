import sys
import importlib

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <0|1|2|3.1|3.2>")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == '0':
        module_name = 'canny_edge_detection_simple'
    elif arg == '1':
        module_name = 'canny_edge_target_matching'
    elif arg == '2':
        module_name = 'yolo_detection_simple'
    elif arg == '3.1':
        module_name = 'match_template_target_select'
    elif arg == '3.2':
        module_name = 'tracker_CSRT_target_select'
    else:
        print("Invalid argument. Please provide 0, 1, or 2.")
        sys.exit(1)
        
    function_name = 'run'
    # Dynamically import the module and call the function
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        func()
    except ImportError:
        print(f"Module {module_name} not found.")
        sys.exit(1)
    except AttributeError:
        print(f"Function {function_name} not found in {module_name}.")
        sys.exit(1)

if __name__ == "__main__":
    main()
