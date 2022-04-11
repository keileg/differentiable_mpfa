import sympy as sym

nc = 3
x = sym.Symbol("x")
p1, p2, A, d1, d2, abyd = sym.symbols("K1 K2 A d1 d2 abyd")
# A, d1, d2 = sym.constants("A d1 d2")
def perm(p):
    return 0.1 + sym.sqrt(p)  # + x ** 2


# perm_from_var = .1 + sym.sqrt(var) #+ var ** 2
# flux = perm * sym.diff(p, x)
# f = sym.diff(flux)


K1 = 0.1 + sym.sqrt(p1)
K2 = 0.1 + sym.sqrt(p2)
K1 = perm(p1)
K2 = perm(p2)
t1 = A * K1 / d1
t2 = A * K2 / d2
t1 = abyd * K1
t2 = abyd * K2

# t1, t2 = sym.symbols("t1 t2")
dval = 0.5 / nc
T12 = 1 / (1 / t1 + 1 / t2)
dTdp1 = sym.diff(T12, p1)
print("dT/dp_1", dTdp1)
print(dTdp1.evalf(subs={p1: 1, p2: 1, d1: dval, d2: dval, A: 1, abyd: 1 / dval}))
dK1dp1 = sym.diff(K1, p1)
print("dK1/dp_1", dK1dp1)
print(dK1dp1.evalf(subs={p1: 1, p2: 1, d1: 0.5 / nc, d2: 0.5 / nc, A: 1}))
t1, t2 = sym.symbols("t1 t2")

tval = 6.6
T12 = 1 / (1 / t1 + 1 / t2)
dTdt1 = sym.diff(T12, t1)
print("dT/dt_1", dTdt1)
print(dTdt1.evalf(subs={t1: tval, t2: tval}))


K1, K2 = sym.symbols("K1 K2")
t1 = abyd * K1
dtdK = sym.diff(t1, K1)
print("dt/dk", dtdK)
print(dtdK.evalf(subs={p1: 1, p2: 1, d1: 0.5 / nc, d2: 0.5 / nc, abyd: 1 / dval}))
