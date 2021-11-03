import random
import string


def id_generator(size=6, chars=string.ascii_uppercase + string.digits) -> str:
    return ''.join(random.choice(chars) for _ in range(size))


if __name__ == "__main__":
    replica_name = f"replica_{id_generator(size=3)}"
    print(replica_name)
