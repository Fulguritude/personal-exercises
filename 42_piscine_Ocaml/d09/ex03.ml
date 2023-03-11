module Try =
	struct
		type 'a t = Success of 'a | Failure of exn ;;
	
		let return (a: 'a): 'a t =
			Success (a)
		;;
	
		let flatten (mma: 'a t t): 'a t =
			match mma with
			| Failure e  -> Failure e
			| Success ma -> ma
		;;
	
		let map (ma: 'a t) (f: 'a -> 'b): 'b t =
			match ma with
			| Failure e -> Failure e
			| Success a -> Success (f (a))
		;;
	
		let bind (ma: 'a t) (mfa: 'a -> 'b t): 'b t =
			flatten (map (ma) (mfa))
		;;
	
		let recover (ma: 'a t)  (mfe: exn -> 'a t): 'a t =
			match ma with
			| Failure e -> mfe (e)
			| Success a -> Success a
		;;
	
		let filter (ma: 'a t) (pred: 'a -> bool): 'a t =
			match ma with
			| Failure e -> Failure e
			| Success a ->
			(
				if pred (a) then Success (a)
				else             Failure (Failure "predicate returned false")
			)
		;;
	end
;;
