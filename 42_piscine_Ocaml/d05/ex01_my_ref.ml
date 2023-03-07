module Ref = struct

	type 'a my_ref = {mutable p: 'a} ;;

	let return (value: 'a): 'a my_ref =
		{ p = value }
	;;

	let get (value: 'a my_ref): 'a =
		let { p = v } = value in
		v
	;;

	let set (pointer: 'a my_ref) (value: 'a): unit =
		pointer.p <- value
	;;

	let bind (pointer: 'a my_ref) (app: ('a -> 'b my_ref)): 'b my_ref =
		app(get(pointer))
	;;

end