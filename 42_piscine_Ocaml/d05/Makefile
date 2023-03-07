EXERCISES := \
	00 \
	01 \
	02 \
	03 \
	04 \
	05 \
	06 \
	07 \
	08 \


all: $(addprefix ex, $(EXERCISES))

ex00:
	ocamlfind ocamlopt -linkpkg -package unix ex00_micronap.ml -o test00 && ./test00 3

ex01:
	ocamlopt ex01_my_ref.ml -o test01 && ./test01

ex02:
	ocamlopt ex02_jokes.ml -o test02 && ./test02

ex03:
	ocamlopt ex03_jokes.ml -o test03 && ./test03 jokes.txt

ex04:
	ocamlopt ex04_sum.ml -o test04 && ./test04

ex05:
	ocamlopt ex05_eu_dist.ml -o test05 && ./test05

ex06:
	ocamlopt ex06_examples_of_file.ml -o test06 && ./test06 ionosphere.csv

ex07:
	ocamlopt ex07_one_nn.ml -o test07 && ./test07 ionosphere.csv

ex08:
	ocamlopt ex08_k_nn.ml -o test08 && ./test08 ionosphere.csv

clean:
	rm *.cmi *.cmx *.o a.out $(addprefix test, $(EXERCISES))
