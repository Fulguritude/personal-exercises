all: ex02

ex02:
	ocamlopt Color.ml Value.ml Card.ml -o a.out

ex03_mli:
	ocamlopt -i Deck.ml > Deck.mli

ex03: ex03_mli
	ocamlopt Color.ml Value.ml Card.ml Deck.ml Deck.mli -o a.out

test: ex03
	ocamlopt  Color.ml Value.ml Card.ml Deck.ml Deck.mli main.ml -o test && ./test

clean:
	rm *.cmi *.cmx *.o a.out test

