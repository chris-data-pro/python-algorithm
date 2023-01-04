if __name__ == '__main__':
    type_order = ['t3', 'c3', 'c5']
    size_order = ['l', 'x', '2', '3', '4', '5', '6', '7', '8', '9']


    def get_key(instance):
        type, size = instance.split('.')
        return type_order.index(type), size_order.index(size[0])

    input1 = ['c3.5xlarge', 'c5.6xlarge', 't3.3xlarge', 'c3.large', 'c5.9xlarge', 'c3.4xlarge']
    sorted_instances = sorted(input1, key=get_key)
    print(sorted_instances)

    input2 = ['c3.large', 'c5.xlarge', 't3.9xlarge', 'c3.xlarge', 'c5.large', 'c3.4xlarge']
    print(sorted(input2, key=get_key))

    input3 = ['c3.large', 'c3.large', 'c5.xlarge', 't3.9xlarge', 'c3.xlarge', 'c5.large', 'c3.4xlarge']
    print(sorted(input3, key=get_key))
