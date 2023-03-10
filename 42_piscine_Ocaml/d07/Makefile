EXERCISES := \
	00 \
	01 \
	02 \
	03 \
	04 \


all: $(addprefix ex, $(EXERCISES))

ex00:
	ocamlopt ex00_people.ml -o test00 && ./test00

ex01:
	ocamlfind ocamlopt -linkpkg -package str ex00_people.ml ex01_doctor.ml -o test01 && ./test01

ex02:
	ocamlfind ocamlopt -linkpkg -package str ex00_people.ml ex01_doctor.ml ex02_dalek.ml -o test02 && ./test02

ex03:
	ocamlfind ocamlopt -linkpkg -package str ex00_people.ml ex01_doctor.ml ex02_dalek.ml ex03_army.ml -o test03 && ./test03

ex04:
	ocamlfind ocamlopt -linkpkg -package str ex00_people.ml ex01_doctor.ml ex02_dalek.ml ex03_army.ml ex04_galifrey.ml -o test04 && ./test04

clean:
	rm *.cmi *.cmx *.o a.out $(addprefix test, $(EXERCISES))
