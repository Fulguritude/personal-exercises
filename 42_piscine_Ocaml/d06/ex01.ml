module HashString = struct
	include String;;

	let hash (s: string): int =
		let get_multiplier(i: int): int =
			match i mod 4 with
			| 1 -> 256
			| 2 -> 256 * 256
			| 3 -> 256 * 256 * 256
			| _ -> 1
		in

		let modulo = 512 in
		let c_list = List.init (String.length s) (String.get (s)) in
		let i_list = List.mapi (fun i c -> int_of_char(c) * get_multiplier(i)) (c_list) in
		let sum    = List.fold_left (+) (0) (i_list) in
		Stdlib.abs(sum) mod modulo
	;;
end

module StringHashtbl = Hashtbl.Make(HashString);;

let () =
	let ht = StringHashtbl.create 5 in
	let values = [ "Hello"; "world"; "42"; "Ocaml"; "H" ] in
	let pairs = List.map (fun s -> (s, String.length s)) values in
	List.iter (fun (k,v) -> StringHashtbl.add ht k v) pairs;
	StringHashtbl.iter (fun k v -> Printf.printf "k = \"%s\", v = %d\n" k v) ht
;;
