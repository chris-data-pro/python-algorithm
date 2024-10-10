import re
import requests


def is_valid_ip(ip):
    # Regular expressions
    p = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
    if p.match(ip):
        parts = ip.split('.')
        return all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)
    return False


if __name__ == '__main__':
    url = "https://www.google.com"
    response = requests.get(url)
    print(response)

    print(response.status_code)
    print(response.url)
    print(response.headers)

    print(is_valid_ip("999.999.999.999"))