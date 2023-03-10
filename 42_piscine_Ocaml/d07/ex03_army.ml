open Ex00_people;;
open Ex01_doctor;;
open Ex02_dalek;;

class ['a] army (x_members: 'a list) =
	object (s)
		val mutable members : 'a list = x_members;

		method add (new_member: 'a): unit =
			members <- new_member :: members
		;

		method remove (): unit =
			match members with
			| []     -> failwith "no members in army"
			| h :: t -> members <- t
		;

		method to_string_list: string list =
			List.map (fun x -> x#to_string) (members)
		;

		method to_string: string =
			let strls = s#to_string_list in
			List.fold_left (fun x y -> x ^ "\n" ^ y) ("") (strls) 
		;

	end
;;

let dalek1 = new dalek;;
let dalek2 = new dalek;;
let dalek3 = new dalek;;

let person1 = new people ("Senshi");;
let person2 = new people ("Marcille");;
let person3 = new people ("Chilchuck");;

let doctor1 = new doctor ("Laios the Cook"     ) (33) (person1);;
let doctor2 = new doctor ("Laios the Erudite"  ) (33) (person2);;
let doctor3 = new doctor ("Laios the Explorer" ) (33) (person3);;

let army1 = new army [dalek1; dalek2;   dalek3];;
let army2 = new army [person1; person2; person3];;
let army3 = new army [doctor1; doctor2; doctor3];;

Printf.printf "Dalek  army: %s\n\n" (army1#to_string);;
Printf.printf "People army: %s\n\n" (army2#to_string);;
Printf.printf "Doctor army: %s\n\n" (army3#to_string);;
