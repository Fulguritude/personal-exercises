EXERCISES := \
	00 \
	01 \
	02 \
	03 \
	04 \
	05 \

LIBS = -linkpkg -package str

EX00 = ex00.ml
EX01 = ex01.ml
EX02 = ex02.ml
EX03 = ex03.ml
EX04 = ex04.ml


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

clean:
	rm *.cmi *.cmx *.o a.out $(addprefix test, $(EXERCISES))
