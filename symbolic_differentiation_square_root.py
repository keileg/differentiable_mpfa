import sympy as sym

x = sym.Symbol("x")
y = sym.Symbol("y")
k0 = sym.Symbol("k0")
two_d = True


p = x * (1 - x)
if two_d:
    p *= y * (1 - y)
perm = sym.sqrt(k0 + p)
flux_x = perm * sym.diff(p, x)
f = sym.diff(flux_x, x)
if two_d:
    flux_y = perm * sym.diff(p, y)
    f += sym.diff(flux_y, y)

print("def source_function(x):\n   return", f)
print("def p_analytical(x):\n   return", p)
