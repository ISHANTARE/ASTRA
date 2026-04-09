import sgp4.api
line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
satrec = sgp4.api.Satrec.twoline2rv(line1, line2)
print("no_kozai (rad/min):", satrec.no_kozai)
print("bstar:", satrec.bstar)
print("inclo (rad):", satrec.inclo)
print("nodeo (rad):", satrec.nodeo)
print("argpo (rad):", satrec.argpo)
print("mo (rad):", satrec.mo)
print("ecco:", satrec.ecco)
print("ndot (rad/min^2):", getattr(satrec, 'ndot', 'Not found'))
print("nddot (rad/min^3):", getattr(satrec, 'nddot', 'Not found'))
