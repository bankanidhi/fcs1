from .correlate import analyse_data_single, find_files
from .args import get_args


def run():
    # Stores the parsed arguments
    options = get_args()
    # Get a list of files that match the keyword as given with the -k arg
    if options.s:
        file_list = find_files(options.s)
        for file in file_list:
            analyse_data_single(file)
    elif options.m:
        file_list = find_files(options.m)

    # If the user wants to perform a dry run.
    if options.list:
        file_list = find_files()
        print("----------------------")
        for f in file_list:
            print(f)
        print("----------------------")
        print("Total files found:", len(file_list))
        print("----------------------")
        exit()

    # # The actually analysis happens after this.
    # elif options.s:
    #     for file in file_list:
    #         analyse_data_single(file)
    # elif options.m:
    #     for file in file_list:
    #         analyse_data_multi(file)
            # generate_report()
