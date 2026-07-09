import json
import ase
from ase.io import read, write

file_read = read('POSCAR', index=':', format='vasp')
# "index"는 얼마 빈도로 데이터를 가져올 것이냐. (ex; 1:1000:10, ::10, :)
# If you want to exchange POSCAR to extxyz file format (not OUTCAR), you must set format="vasp".

for structure in file_read:
    write('output.extxyz', structure, format="extxyz", append=True)
    # append = True로 해둬야 파일이 덮어써지지 않고 추가되어 써집니다. 파이썬 write의 "a".

