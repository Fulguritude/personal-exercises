type phospate = string;;
type deoxyribose = string;;
type nucleobase = A | T | C | G | None;;

type nucleotide =
{
	phospate    : phospate;
	deoxyribose : deoxyribose;
	nucleobase  : nucleobase;
};;

let generate_nucleotide (c: char): nucleotide =
	{
		phospate    = "phospate";
		deoxyribose = "deoxyribose";
		nucleobase  =
			match c with
			| 'A' -> A
			| 'T' -> T
			| 'C' -> C
			| 'G' -> G
			|  _  -> None
	}
;;

let char_of_nucleotide (n: nucleotide): char =
	match n.nucleobase with
	| A -> 'A'
	| T -> 'T'
	| C -> 'C'
	| G -> 'G'
	| _ -> 'x'
;;

let string_of_nucleotide (n: nucleotide): string =
	match n.nucleobase with
	| A -> "A"
	| T -> "T"
	| C -> "C"
	| G -> "G"
	| _ -> "x"
;;


type helix = nucleotide list;;

let rec generate_helix (n: int): helix =
	Random.self_init ();
	let get_random_char (unit): char =
		(* doc says 'bound' arg is exclusive, but LSP says that 4 isn't matched *)
		match Random.int 4 with
		| 0 -> 'A'
		| 1 -> 'T'
		| 2 -> 'C'
		| 3 -> 'G'
		| _ -> assert false
	in

	match n with
	| 0 -> []
	| n when n > 0 ->
	(
		let new_nucleotide = generate_nucleotide ( get_random_char () ) in
		new_nucleotide :: generate_helix (n - 1)
	)
	| _ -> []
;;

let rec string_of_helix (h : helix): string =
	match h with
	| [] -> ""
	| hlx_h :: hlx_t -> string_of_nucleotide(hlx_h) ^ string_of_helix(hlx_t)
;;


let rec complementary_helix (h: helix): helix =
	match h with
	| [] -> []
	| hlx_h :: hlx_t ->
	(
		let char_old = char_of_nucleotide(hlx_h) in
		let char_new =
			match char_old with
			| 'A' -> 'T'
			| 'T' -> 'A'
			| 'C' -> 'G'
			| 'G' -> 'C'
			| _   -> 'x'
		in
		let paired_nucleotide = generate_nucleotide(char_new) in
		paired_nucleotide :: complementary_helix (hlx_t)
	)
;;



let hlx_test = generate_helix (10) ;;
let hlx_pair = complementary_helix (hlx_test) ;;

print_endline(string_of_helix(hlx_test));
print_endline(string_of_helix(hlx_pair));
