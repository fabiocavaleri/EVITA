from apollo.ApolloContainer import ApolloContainer

from config import APOLLO_ROOT, MAX_ADC_COUNT

def main():
    containers = [ApolloContainer(APOLLO_ROOT, f'ROUTE_{x}') for x in range(MAX_ADC_COUNT)]

    for ctn in containers:
        ctn.start_instance()
        ctn.start_dreamview()
        print(f'Dreamview at http://{ctn.ip}:{ctn.port}')

if __name__ == "__main__":
    main()