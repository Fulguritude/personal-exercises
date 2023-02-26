class Account(object):

	ID_COUNT = 0

	def __init__(self, name, **kwargs):
		self.id = self.ID_COUNT 
		self.name = name 
		self.__dict__.update(kwargs) 
		if 'value' in kwargs:
        	self.value = 0
        self.ID_COUNT += 1

	def transfer(self, amount):
		self.value += amount

	def is_corrupted(self):
		attributes = dir(self)
		if (len(attributes) % 2 == 0)
			return True
		if (not 'name' is in attributes or
				not 'id' is in attributes or
				not 'value' is in attributes)
			return True
		for s_att in attributes:
			length = len(s_att)
			if length >= 1 and s_att[0] == 'b':
				return True
			if length >= 3 and s_att[0:3] == 'zip':
				return True
			if length >= 4 and s_att[0:4] == 'addr':
				return True
		return False



class Bank(object):
	"""The bank"""

	def __init__(self):
		self.account_lst = []

	def __get_account(self, id_or_str):
		if isinstance(id_or_str, int):
			for acc in self.account_lst:
				if id_or_str == acc.id:
					return acc
			print("Account not found through id.")
			return None
		elif isinstance(id_or_str, str):
			for acc in self.account_lst:
				if id_or_str == acc.name:
					return acc
			print("Account not found through id.")
			return None
		else:
			print("Invalid account identifier.")
		return None

	def add(self, account):
		self.account.append(account)

	def transfer(self, origin, dest, amount):
		"""
			@origin:  int(id) or str(name) of the first account
			@dest:    int(id) or str(name) of the destination account
			@amount:  float(amount) amount to transfer
			@return         True is success, False is an error occured
		"""
		creditor = self.__get_account(origin)
		if creditor is None:
			print("Invalid identifier for creditor")
			return False
		debtor = self.__get_account(dest)
		if debtor is None:
			print("Invalid identifier for debtor")
			return False
		if creditor.is_corrupted():
			print("Creditor account is corrupted")
			return False
		if debtor.is_corrupted():
			print("Debtor account is corrupted")
			return False
		if amount <= 0:
			print("Invalid transfer amount")
			return False
		if amount > creditor.value:
			print("Not enough funds in creditor account")
			return False
		creditor.transfer(-amount)
		debtor.transfer(amount)
		return True


	def fix_account(self, account):
		"""
			fix the corrupted account
			@account: int(id) or str(name) of the account
			@return         True is success, False is an error occured
		"""