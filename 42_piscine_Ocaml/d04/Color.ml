module Color = struct

	type t = Spade | Heart | Diamond | Club ;;


	let all (): t list =
		[Spade; Heart; Diamond; Club]
	;;


	let toString (color: t): string =
		match color with
		| Spade   -> "S"
		| Heart   -> "H"
		| Diamond -> "D"
		| Club    -> "C"
	;;


	let toStringVerbose (color: t): string =
		match color with
		| Spade   -> "Spade"
		| Heart   -> "Heart"
		| Diamond -> "Diamond"
		| Club    -> "Club"
	;;


	(* Going for the bridge order ♠ > ♥ > ♦ > ♣ *)
	let toInt (color: t): int =
		match color with
		| Spade   -> 4
		| Heart   -> 3
		| Diamond -> 2
		| Club    -> 1
	;;
end
