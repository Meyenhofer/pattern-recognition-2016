from time import asctime
from timeit import default_timer as Timer
from molecules import molecule


def main():
    print("%s | Start running molecules_pipeline." % asctime())
    print("================================================================================")
    start = Timer()
    
    test = molecule.Molecule("16")
    print(test.get_id())
    print("Nodes:")
    for node in test.get_nodes():
        print(node)
    
    end = Timer()
    print("================================================================================")
    print("Duration: %f" % (end - start))
    print("%s | End of molecules_pipeline." % asctime())


# Program entry point. Don't execute if imported.
if __name__ == '__main__':
    main()