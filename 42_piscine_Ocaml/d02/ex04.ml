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
			| _ -> None
	}
;;

