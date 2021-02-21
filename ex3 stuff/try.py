

def choose(i):
    if i == 12 or i == 0:
        return 1
    return atzeret(12) / (atzeret(12 - i) * atzeret(i))


def atzeret(n):
    sum = n
    for i in range(2, n):
        sum *= i
    return sum


ig = 0.65
sum = 0
for i in range(9, 13):
    a = choose(i)
    p = (0.8 ** i)
    p_c = (0.2 ** (12 - i))
    sum += a * p * p_c
sum *= ig
ii = 0.35
sum1 = 0
for i in range(9):
    a = choose(i)
    p = (0.1 ** i)
    p_c = (0.9 ** (12 - i))
    sum1 += a * p * p_c
sum1 *= ii
print("result: ", sum + sum1)
