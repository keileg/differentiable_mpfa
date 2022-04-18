import sympy as sym

x = sym.Symbol("x")
y = sym.Symbol("y")
var = sym.Symbol("var")

two_d = True


p = x * (1 - x)
if two_d:
    p *= y * (1 - y)
perm = 0.1 + x**2
perm_from_var = 0.1 + var**2
flux_x = perm * sym.diff(p, x)
f = sym.diff(flux_x, x)
if two_d:
    flux_y = perm * sym.diff(p, y)
    f += sym.diff(flux_y, y)

print("def source_function(x):\n   return", f)
print("def p_analytical(x):\n   return", p)
print("def permeability_from_pressure(var):\n   return", perm_from_var)
