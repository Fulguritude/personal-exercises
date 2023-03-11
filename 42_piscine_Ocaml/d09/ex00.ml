module Watchtower =
	struct
		type hour = int;;

		let zero : hour = 0;;

		let add (h1: hour) (h2: hour): hour =
			let result = (h1 + h2) mod 12 in
			if result < 0 then result + 12 else result
		;;

		let sub (h1: hour) (h2: hour): hour =
			let result = (h1 - h2) mod 12 in
			if result < 0 then result + 12 else result
		;;

	end
;;

assert ( Watchtower.add (10) ( 10) =  8);;
assert ( Watchtower.sub (10) ( 12) = 10);;
assert ( Watchtower.add (10) (-11) = 11);;
assert ( Watchtower.sub (10) (-22) =  8);;
