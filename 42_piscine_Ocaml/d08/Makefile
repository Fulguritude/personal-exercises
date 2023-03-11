EXERCISES := \
	00 \
	01 \
	02 \
	03 \
	04 \
	05 \

LIBS = -linkpkg -package str

EX00=        ex00_atom.ml
EX01=$(EX00) ex01_molecules.ml
EX02=$(EX01) ex02_alkanes.ml
EX03=$(EX02) ex03_reactions.ml
EX04=$(EX03) ex04_alkane_combustion.ml
EX05=$(EX04) ex05_incomplete_combustion.ml


all: $(addprefix ex, $(EXERCISES))

ex00:
	ocamlfind ocamlopt $(LIBS) $(EX00) -o test00 && ./test00

ex01:
	ocamlfind ocamlopt $(LIBS) $(EX01) -o test01 && ./test01

ex02:
	ocamlfind ocamlopt $(LIBS) $(EX02) -o test02 && ./test02

ex03:
	ocamlfind ocamlopt $(LIBS) $(EX03) -o test03 && ./test03

ex04:
	ocamlfind ocamlopt $(LIBS) $(EX04) -o test04 && ./test04

ex05:
	ocamlfind ocamlopt $(LIBS) $(EX05) -o test05 && ./test05

clean:
	rm *.cmi *.cmx *.o a.out $(addprefix test, $(EXERCISES))
