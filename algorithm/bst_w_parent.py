class Employee:
    def __init__(self, name, employee_id, manager):
        self.name = name
        self.employee_id = employee_id
        self.manager = manager


def find_the_reporting_chain(e: Employee) -> list:
    reporting_chain = []

    while e.manager is not None:
        reporting_chain += [e.name]
        e = e.manager

    reporting_chain += ["CEO"]
    return reporting_chain


def inorder_traverse(res, node):
    if not node:
        return []

    res.append(node.name)
    res += inorder_traverse(res, node.manager)
    return res


def find_the_reporting_chain_1(e: Employee):
    reporting_chain = []
    return inorder_traverse(reporting_chain, e)


def main():
    ceo = Employee("CEO", 1, None)
    director1 = Employee("D1", 2, ceo)
    manager1 = Employee("M1", 3, director1)
    manager2 = Employee("M2", 4, director1)
    ic1 = Employee("IC1", 5, manager1)
    ic2 = Employee("IC2", 6, manager1)
    ic3 = Employee("IC3", 7, manager2)
    reporting_chain_1 = find_the_reporting_chain(ic3)
    reporting_chain_2 = find_the_reporting_chain(director1)
    # Assert the reporting chain is ["IC3", "M2", "D1", "CEO"]
    assert reporting_chain_1 == ["IC3", "M2", "D1", "CEO"]
    # Assert the reporting chain is ["D1", "CEO"]
    assert reporting_chain_2 == ["D1", "CEO"]
    print(find_the_reporting_chain(ic3))
    print(find_the_reporting_chain(director1))
    print(find_the_reporting_chain_1(ic3))
    print(find_the_reporting_chain_1(ceo))


if __name__ == '__main__':
    main()
