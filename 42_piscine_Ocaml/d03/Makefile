all: ex00 ex01

ex00:
	ocamlfind ocamlopt -linkpkg -package graphics -package unix ex00_graphics.ml -o test00

ex01:
	ocamlfind ocamlopt -linkpkg -package graphics -package unix ex00_graphics.ml ex01_gardening.ml -o test01

run: all
	./$(EXEC_NAME)

clean:
	rm *.cmi *.cmx *.o a.out test00 test01

