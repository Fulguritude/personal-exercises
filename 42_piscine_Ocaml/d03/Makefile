all: ex00 ex01 ex02 ex03

ex00:
	ocamlfind ocamlopt -linkpkg -package graphics -package unix ex00_graphics.ml -o test00

ex01:
	ocamlfind ocamlopt -linkpkg -package graphics -package unix ex00_graphics.ml ex01_gardening.ml -o test01

ex02:
	ocamlfind ocamlopt ex02_cipher.ml -o test02

ex03:
	ocamlfind ocamlopt -linkpkg -package graphics -package unix ex00_graphics.ml ex01_gardening.ml ex03_btree.ml -o test03

run: all
	./$(EXEC_NAME)

clean:
	rm *.cmi *.cmx *.o a.out test00 test01 test02 test03

