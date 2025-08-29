import csv

print("Welcome to Phonebook")

def main():
    while True:
        printMenu()
        selectedOption = int(input("Enter an option: "))
        if (selectedOption == 1):
            printAllPhoneNumbers()
        elif (selectedOption == 2):
            storePhoneNumber()
        elif (selectedOption == 3):
            exit()
        else:
            print("Invalid Option")
            main()


def printMenu():
    print("1. Print all Phone numbers")
    print("2. Store A New Phone Number")
    print("3. Exit")

def storePhoneNumber():
    name = input("Enter name: ")
    phone = input("Enter phone number: ")

    with open("phoneNumbers.csv", mode='+a', newline="\n") as file:
        writer = csv.writer(file)
        writer.writerow([name,phone])
        file.close()
        print("Number Saved!!")
def printAllPhoneNumbers():
    with open("phoneNumbers.csv",mode='r', newline="\n") as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)
        

main()