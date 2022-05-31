import sympy as sym

x = sym.Symbol("x")
y = sym.Symbol("y")
var = sym.Symbol("var")
k = sym.Symbol("k0")
k1 = sym.Symbol("k1")
p0 = sym.Symbol("p0")
two_d = True


p = x**2  # (1 - x)
# if two_d:
#     p *= y ** 2 # (1 - y)
p *= k1
p += p0
perm = sym.sqrt(k + p)  # + x ** 2
perm_from_var = sym.sqrt(k + var)  # + var ** 2
flux_x = perm * sym.diff(p, x)
f = sym.diff(flux_x, x)
if two_d:
    flux_y = perm * sym.diff(p, y)
    f += sym.diff(flux_y, y)

print("def source_function(x):\n   return", f)
print("def p_analytical(x):\n   return", p)
print("def permeability_from_pressure(var):\n   return", perm_from_var)
