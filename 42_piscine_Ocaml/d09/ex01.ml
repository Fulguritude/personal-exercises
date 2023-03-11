module App =
	struct
		type project = string * string * int ;;

		let zero : project = ("", "", 0);;

		let combine (p1: project) (p2: project): project =
			let (s1, _, grade1) = p1 in
			let (s2, _, grade2) = p2 in
			let new_grade  = (grade1 + grade2) / 2 in
			let new_str    = s1 ^ s2 in
			let new_status = if new_grade > 80 then "success" else "failure" in
			(new_str, new_status, new_grade)
		;;

		let fail (p: project): project =
			let (s, _, _) = p in
			(s, "failure", 0)
		;;

		let succeed (p: project): project =
			let (s, _, _) = p in
			(s, "success", 80)
		;;

		let string_of_project (p: project): string =
			let (str, stat, grade) = p in
			Printf.sprintf "(%s, %s, %d)" str stat grade
		;;

		let print_proj (p: project): unit =
			print_endline(string_of_project(p))
		;;
	end
;;

App.print_proj(App.combine ("test1", "success", 95) (App.succeed(App.zero)) );;
