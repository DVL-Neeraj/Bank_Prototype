from random import randint

from .Customer import Customer
from .helpers import write_output_to_csv
import os


def simulate(number_of_customers=int(1e7), new_base_data=False):
    """Simulate a customer dataset and store it in a csv file"""

    if new_base_data == True:
        file_path = "data/output/customers.csv"
        temp_file_path = "data/output/temp.csv"
        customers = []

        for i in range(number_of_customers):
            customer = Customer()
            customers.append(customer)
            if i % 1000 == 0:
                print(f"{i / number_of_customers * 100:.1f}%")
                write_output_to_csv(temp_file_path, customers)
                customers = []
        write_output_to_csv(temp_file_path, customers)

        if os.path.isfile("file_path"):
            os.remove(file_path)
        os.rename(temp_file_path, file_path)
    else:
        return
