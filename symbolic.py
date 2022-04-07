import sympy as sym

x = sym.Symbol('x')
var = sym.Symbol('var')

p = x * (1-x)
perm = .1 + sym.sqrt(p) #+ x ** 2
perm_from_var = .1 + sym.sqrt(var) #+ var ** 2
flux = perm * sym.diff(p, x)
f = sym.diff(flux)
print("def source_function(x):\n   return", f)
print("def p_analytical(x):\n   return", p)
print("def permeability_from_pressure(var):\n   return", perm_from_var)