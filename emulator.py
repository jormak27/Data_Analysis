# Tryo Bytes: Jordan Makansi, Florian Fontaine-Papion, Michael Fratoni
# CEE 290I
# dlx emulator

## Initialize program  by creating a compiler object##

''' PROGRAM COUNTER POINTS TO INSTRUCTION REGISTERS

each instruction register is one byte (or 8 bits)

So, the pc increments by 4 after each instruction because it wants to jump 32 bits, or 4 bytes. 
'''


## TO DO:
'''
- finish writing the instructions function (at the end)
- finish writing methods for Compiler class
- add an SP or stack pointer '''

### PARAMETERS OF MACHINE ####

N_REGISTERS = 8
RAM = 8

class Compiler:

    def __init__(self):
        self.pc = 0   # program counter.  points to instruction register (see below) 
        self.memory = [None]*RAM
        self.ir = self.memory[self.pc]  # stands for instruction register. this is a single register  
        self.registers = [None]*N_REGISTERS    #these registers can be thought of as on the stack for the purposes of this program
        self.registers[0] = 0
        
    def __str__(self):
        string = '' 
        string += 'PROGRAM COUNTER: ' + str(self.pc) + '\n\n'
        string += 'INSTRUCTION REGISTER: \n' + str(self.ir) + '\n\n'
        string += 'REGISTER: \n' + str(self.print_registers()) + '\n\n'
        string += 'MEMORY: \n' + str(self.print_memory()) + '\n\n'
        return string

    def print_memory(self):
        string = ''
        for i in self.memory:
            string = string + str(i) + '\n'  
        return string

    def print_registers(self):
        string = ''
        for i in self.registers:
            string = string + str(i) +'\n'
        return string

    def execute_instructions(self):
        '''we are assuming that the only thing in memory is the instructions.'''
        '''base case if instruction register is empty'''
        instructions = self.memory
        if instructions==[]:
            print 'Do nothing if instruction register is empty'
        else:
            for instruction in instructions:
                if instruction == None:
                    exit 
                else:
                    operation(instruction,self)

def parse_command(command):
    '''This function takes in a command in the form of a string from a text file.
    output is four parameters:
        op, A, B, and C'''
    op=command.split()[0]  # splits the command at the white space and stores first element
    A,B,C = command.split()[1].split(',')
    ir = [op, A, B, C] #ir stands for Instruction Register 
    return ir

def read_text_file():
    text_file = open('TEST.txt', 'r')
    commands = text_file.readlines()
    instructions = []
    for command in commands:
        ir = parse_command(command)
        instructions += [ir] 
    return instructions   #this comes out as a list of instructions

def operation(instruction,Compiler):
    ''' this function takes in a command as a string,
    and executes that command.
    sample instruction = ['ADD', '1','2','3']
    '''
    reg = Compiler.registers # this is an array of registers
    op = instruction[0]
    A = int(instruction[1]) # convert these strings into integers so that we can use them to index the memory
    B = int(instruction[2])
    C = int(instruction[3])
    mem = Compiler.memory
    if op=='ADDI':
        reg[A] = reg[B] + C
        Compiler.pc += 1   ## increment pc by 4 (depeneding on the length of the address)
        print 'executed the add command'
    elif op=='SUBI':
        reg[A] = reg[B] - C
        Compiler.pc += 1 ## increment pc by 4 (depeneding on the length of the address)
    elif op=='MULI':
        reg[A] = reg[B] * C
        Compiler.pc += 1   ## increment pc by 4 (depeneding on the length of the address)
    elif op=='DIVI':
        reg[A] = reg[B]/C
        Compiler.pc += 1  ## increment pc by 4 (depeneding on the length of the address)
    elif op == 'LDW':
        reg[A] = mem[(reg[B]+C)/1]
        ## this loads in register A the value in register b, plus register c 
        ## increment pc by 4 (depeneding on the length of the address)
    else :
        pass

def main():
    Tryo_Bytes = Compiler()
    instructions = read_text_file()
    #load the text file into the CPU memory.  each element in the list "memory" holds an instruction
    for i in range(len(instructions)):
        Tryo_Bytes.memory[i] = instructions[i]  #load instructions into memory
    Tryo_Bytes.execute_instructions()
    print Tryo_Bytes
    #execute instructions one by one
    return

main()



