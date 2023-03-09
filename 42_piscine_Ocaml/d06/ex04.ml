module type VAL = sig
	type t

	val add : t -> t -> t
	val mul : t -> t -> t
end
