EXERCISES := \
	00 \
	01 \
	02 \
	03 \
	04 \


all: $(addprefix ex, $(EXERCISES))



ex00:
	ocamlopt ex00.ml -o test00 && ./test00

ex01:
	ocamlopt ex01.ml -o test01 && ./test01

ex02:
	ocamlopt ex02.ml -o test02 && ./test02

ex03:
	ocamlopt ex03.ml -o test03 && ./test03

ex04:
	ocamlopt ex04.ml -o test04 && ./test04



clean:
	rm *.cmi *.cmx *.o a.out $(addprefix test, $(EXERCISES))
