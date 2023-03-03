type phospate = string;;
type deoxyribose = string;;
type nucleobase = A | T | C | G | U | None;;

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
			| 'U' -> U
			|  _  -> None
	}
;;

let char_of_nucleotide (n: nucleotide): char =
	match n.nucleobase with
	| A -> 'A'
	| T -> 'T'
	| C -> 'C'
	| G -> 'G'
	| U -> 'U'
	| _ -> 'x'
;;

let string_of_nucleotide (n: nucleotide): string =
	match n.nucleobase with
	| A -> "A"
	| T -> "T"
	| C -> "C"
	| G -> "G"
	| U -> "U"
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



type rna = nucleotide list;;

(*
	If I were less lazy, or if this was more important I would refactor to
	have generate_rna and complementary_helix both be partial applications
	of a common function, where only the output char for 'A' is missing.
	However, given that the subject already requires us to copy-paste code,
	the requirement for DRY code aren't that high...
*)
let rec generate_rna (h: helix): rna =
	match h with
	| [] -> []
	| hlx_h :: hlx_t ->
	(
		let char_old = char_of_nucleotide(hlx_h) in
		let char_new =
			match char_old with
			| 'A' -> 'U'
			| 'T' -> 'A'
			| 'C' -> 'G'
			| 'G' -> 'C'
			| _   -> 'x'
		in
		let paired_nucleotide = generate_nucleotide(char_new) in
		paired_nucleotide :: generate_rna (hlx_t)
	)
;;



type nucleotide_triplet = nucleotide * nucleotide * nucleotide ;;

type aminoacid =
	| Stop
	| Ala
	| Arg
	| Asn
	| Asp
	| Cys
	| Gln
	| Glu
	| Gly
	| His
	| Ile
	| Leu
	| Lys
	| Met
	| Phe
	| Pro
	| Ser
	| Thr
	| Trp
	| Tyr
	| Val
;;

type protein = aminoacid list;;


let name_of_aminoacid (codon: aminoacid): string =
	match codon with
	| Stop -> "End"
	| Ala  -> "Alanine"
	| Arg  -> "Arginine"
	| Asn  -> "Asparagine"
	| Asp  -> "Aspartique"
	| Cys  -> "Cysteine"
	| Gln  -> "Glutamine"
	| Glu  -> "Glutamique"
	| Gly  -> "Glycine"
	| His  -> "Histidine"
	| Ile  -> "Isoleucine"
	| Leu  -> "Leucine"
	| Lys  -> "Lysine"
	| Met  -> "Methionine"
	| Phe  -> "Phenylalanine"
	| Pro  -> "Proline"
	| Ser  -> "Serine"
	| Thr  -> "Threonine"
	| Trp  -> "Tryptophane"
	| Tyr  -> "Tyrosine"
	| Val  -> "Valine"
;;

let string_of_aminoacid (codon: aminoacid): string =
	match codon with
	| Stop -> "Stop"
	| Ala  -> "Ala"
	| Arg  -> "Arg"
	| Asn  -> "Asn"
	| Asp  -> "Asp"
	| Cys  -> "Cys"
	| Gln  -> "Gln"
	| Glu  -> "Glu"
	| Gly  -> "Gly"
	| His  -> "His"
	| Ile  -> "Ile"
	| Leu  -> "Leu"
	| Lys  -> "Lys"
	| Met  -> "Met"
	| Phe  -> "Phe"
	| Pro  -> "Pro"
	| Ser  -> "Ser"
	| Thr  -> "Thr"
	| Trp  -> "Trp"
	| Tyr  -> "Tyr"
	| Val  -> "Val"
;;

let aminoacid_of_string (s: string): aminoacid =
	match s with
	| "UAA" | "UAG" | "UGA"                         -> Stop
	| "GCA" | "GCC" | "GCG" | "GCU"                 -> Ala
	| "AGA" | "AGG" | "CGA" | "CGC" | "CGG" | "CGU" -> Arg
	| "AAC" | "AAU"                                 -> Asn
	| "GAC" | "GAU"                                 -> Asp
	| "UGC" | "UGU"                                 -> Cys
	| "CAA" | "CAG"                                 -> Gln
	| "GAA" | "GAG"                                 -> Glu
	| "GGA" | "GGC" | "GGG" | "GGU"                 -> Gly
	| "CAC" | "CAU"                                 -> His
	| "AUA" | "AUC" | "AUU"                         -> Ile
	| "CUA" | "CUC" | "CUG" | "CUU" | "UUA" | "UUG" -> Leu
	| "AAA" | "AAG"                                 -> Lys
	| "AUG"                                         -> Met
	| "UUC" | "UUU"                                 -> Phe
	| "CCC" | "CCA" | "CCG" | "CCU"                 -> Pro
	| "UCA" | "UCC" | "UCG" | "UCU" | "AGU" | "AGC" -> Ser
	| "ACA" | "ACC" | "ACG" | "ACU"                 -> Thr
	| "UGG"                                         -> Trp
	| "UAC" | "UAU"                                 -> Tyr
	| "GUA" | "GUC" | "GUG" | "GUU"                 -> Val
	| _ -> assert false (* crash if inconsistent impl*)
;;


let rec string_of_protein (prot: protein): string =
	match prot with
	| h :: t -> string_of_aminoacid (h) (* ^ " " *) ^ string_of_protein (t)
	| []     -> ""
;;


let rec generate_base_triplets (strand: rna): nucleotide_triplet list =
	match strand with
	| base1 :: base2 :: base3 :: remainder ->
	(
		(base1, base2, base3) :: generate_base_triplets (remainder)
	)
	| _ -> []
;;



let decode_rna (strand: rna): protein =
	(* subject doesn't ask to handle the "no 'Stop' codon" case *)
	let aminoacid_of_triplet (t: nucleotide_triplet): aminoacid =
		match t with | (base1, base2, base3) ->
		let str1  = string_of_nucleotide (base1) in
		let str2  = string_of_nucleotide (base2) in
		let str3  = string_of_nucleotide (base3) in
		let codon = str1 ^ str2 ^ str3           in
		aminoacid_of_string(codon)
	in

	let triplets   = generate_base_triplets (strand) in
	let aminoacids = List.map aminoacid_of_triplet triplets in
	let rec list_cutoff_at_filter (f: ('a -> bool)) (l: 'a list): 'a list =
		match l with
		| [] -> []
		| h :: t ->
		(
			if f (h) then
				[]
			else
				h :: list_cutoff_at_filter (f) (t) 
		)
	in
	list_cutoff_at_filter (fun x -> x = Stop) aminoacids
;;



let helix_of_string (s: string): helix =
	let l = List.init (String.length s) (String.get s) in
	List.map (generate_nucleotide) (l)
;;

let life (s: string): protein =
	print_endline(s                       ); let helix = helix_of_string (s)     in
	print_endline(string_of_helix(helix)  ); let rna   = generate_rna    (helix) in
	print_endline(string_of_helix(rna)    ); let prot  = decode_rna      (rna)   in
	print_endline(string_of_protein(prot) );
	prot
;;

let result = life ("CGTTCTTTGATTAAAAAAA") ;; (* Ala Arg Asn *)
