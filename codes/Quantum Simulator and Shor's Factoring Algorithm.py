import numpy as np
import matplotlib.pyplot as plt
import timeit
from fractions import Fraction
import math

#Dirac Notation

def PrettyPrintBinary(myState):
    print("( ", end="")
    i = 0
    for state in myState:
        print(state[0], "|" + str(state[1]) + ">", end="")
        if i < len(myState) - 1:
            print(" + ", end="")
        i += 1
    print(" )")

def PrettyPrintInteger(myState):
    print("( ", end="")
    i = 0
    for state in myState:
        print(state[0], "|" + str(int(str(state[1]),2)) + ">", end="")
        if i < len(myState) - 1:
            print(" + ", end="")
        i += 1
    print(" )")


def DiracToVec(myState):
    num_bits = len(myState[0][1])
    vec = [0 for i in range(2**num_bits)]
    for state in myState:
        index = int(state[1],2)
        vec[index] += np.round(state[0],3)
    return vec

def VecToDirac(myVec):
    state = []
    num_bits = int(np.log2(len(myVec)))
    for i in range(len(myVec)):
        elem = myVec[i]
        if elem != 0:
            state.append((elem, format(i, "0" + str(num_bits) + "b")))
    return state


#Reading Input String from File

def ReadInputString(myInput_lines):
    myInput=[]
    myInput_lines=myInput_lines.split('\n')
    myInput_lines = [ i for i in myInput_lines if i!='']
    num_wires=int(myInput_lines[0])
    measure = False
    myState = [1 if i == 0 else 0 for i in range(2**num_wires)]
    for line in myInput_lines[1:]:
        if len(line.split()) == 0:
            continue
        myInput.append(line.split())
        if line == "MEASURE":
            measure = True
        elif line.split()[0] == "INITSTATE":
            if line.split()[1] == "FILE":
                myState = Readinput_stateFromFile(line.split()[2])
            elif line.split()[1] == "BASIS":
                myState = Readinput_stateFromBasis(line.split()[2])
    return (num_wires,myInput,myState, measure)

def Readinput_stateFromFile(file):
    input_lines = open(file).read()
    input_lines = input_lines.split('\n')
    input_lines = [ i for i in input_lines if i!='']

    state = []
    for line in input_lines:
        real = float(line.split()[0])
        imag = float(line.split()[1])
        state.append(real + 1j*imag)

    return state

def Readinput_stateFromBasis(basis):
    index = int(basis[1:(len(basis)-1)], 2)

    myState = [1 if i == index else 0 for i in range(2**(len(basis)-2))]

    return myState

def precompile(circuit):
    for i, gate in enumerate(circuit):
        circuit[i] = [eval(val) if "pi" in val else val for val in gate]
    
    depth = 1
    j = 0
    while j < depth:
        i = 0
        while i < len(circuit):
            gate = circuit[i]
            name = gate[0]
            
            if name == "CNOT":
                control = circuit[i][1]
                output = circuit[i][2]

                if abs(int(control) - int(output)) > 1: # Long Range
                    tempgate = "0"
                    if int(control) > int(output):
                        tempgate = str(int(output)+1)
                    else:
                        tempgate = str(int(control)+1)
                    
                    circuit[i] = ["SWAP", tempgate, control]
                    circuit.insert(i+1, ["CNOT", tempgate, output])
                    circuit.insert(i+2, ["SWAP", tempgate, control])
                    depth += 1
            
            elif name == "NOT":
                gate = gate[1]
                circuit[i:i+1] = [
                    ["H", gate],
                    ["P", gate, str(np.pi)],
                    ["H", gate],
                ]
            
            elif name == "RZ":
                gate, phase = gate[1], float(gate[2])
                circuit[i:i+1] = [
                    ["NOT", gate],
                    ["P", gate, str(-phase / 2)],
                    ["NOT", gate],
                    ["P", gate, str(phase / 2)],
                ]
                depth += 1
            
            elif name == "CRZ":
                control, output, phase = gate[1], gate[2], float(gate[3])
                circuit[i:i+1] = [
                    ["CNOT", control, output],
                    ["P", output, str(-phase / 2)],
                    ["CNOT", control, output],
                    ["P", output, str(phase / 2)],
                ]
                depth += 1
            
            elif name == "CPHASE":
                control, output, phase = gate[1], gate[2], float(gate[3])
                circuit[i:i+1] = [
                    ["CRZ", control, output, str(phase)],
                    ["P", control, str(phase / 2)],
                ]
                depth += 1
            
            elif name == "SWAP":
                gate1 = circuit[i][1]
                gate2 = circuit[i][2]

                count = 0
                if abs(int(gate1) - int(gate2)) > 1:
                    for g in range(int(gate1), int(gate2)):
                        if count == 0:
                            circuit[i] = ["SWAP", str(g), str(g+1)]
                        else:
                            circuit.insert(i+count, ["SWAP", str(g), str(g+1)])
                        count += 1
                    for g in range(int(gate2)-1, int(gate1), -1):
                        circuit.insert(i+count, ["SWAP", str(g), str(g-1)])
                        count += 1
                    depth += 1
                else:
                    circuit[i] = ["CNOT", gate1, gate2]
                    circuit.insert(i+1, ["CNOT", gate2, gate1])
                    circuit.insert(i+2, ["CNOT", gate1, gate2])
        
            i += 1
        j += 1
    
    return circuit

#Measurement

def measureState(myState):
    myStateConj = np.conjugate(myState)
    myState = np.real(np.multiply(myStateConj, myState))

    return "|" + format(np.where(np.random.multinomial(1, myState) == 1)[0][0], "0" + str(int(np.log2(len(myState)))) + "b") + ">"

#Simulator S

def H(wire,input_state):
  new_state = []

  # Iterate through each element in the input state
  for element in input_state:
    if int(element[1][wire]) == 0:
      # If the qubit is in state |0> then output (|0> + |1>)/sqrt(2)
      first = element[1][:wire] + "0" + element[1][wire+1:]
      second = element[1][:wire] + "1" + element[1][wire+1:]
      new_state.append(((1/np.sqrt(2))*element[0], first))
      new_state.append(((1/np.sqrt(2))*element[0], second))
    if int(element[1][wire]) == 1:
      # If the qubit is in state |0> then output (|0> - |1>)/sqrt(2)
      first = element[1][:wire] + "0" + element[1][wire+1:]
      second = element[1][:wire] + "1" + element[1][wire+1:]
      new_state.append(((1/np.sqrt(2))*element[0], first))
      new_state.append((-(1/np.sqrt(2))*element[0], second))
  return new_state


def P(wire, theta, input_state):
    new_state = []

    # Iterate through each element in the input state
    for element in input_state:
        amplitude, basis = element  # Unpack amplitude and basis state
        if int(basis[wire]) == 0:
            # If the qubit is in state |0>, no phase change
            new_state.append((amplitude, basis))
        elif int(basis[wire]) == 1:
            # If the qubit is in state |1>, apply phase shift e^(i * theta)
            newAmplitude = amplitude * np.exp(1j * theta)
            new_state.append((newAmplitude, basis))

    return new_state


def CNOT(controlWire,notWire,input_state):
    new_state = []

    # Iterate through each element in the input state
    for element in input_state:
        if int(element[1][controlWire]) == 0:
          # If the qubit is in state |0> do nothing
          new_state.append((element[0], element[1]))
        elif int(element[1][controlWire]) == 1:
          # If the qubit is in state |1> flip the other qubit the gate is acting on
          changed = element[1][:notWire] + ("1" if element[1][notWire] == "0" else "0") + element[1][notWire+1:]
          new_state.append((element[0], changed))
    return new_state

def xyModN(first_wire, num_wires, x, N, input_state):
    new_state = []

    for element in input_state:
        amplitude, basis = element
        binary = basis[first_wire:(first_wire+num_wires)]
        left = basis[:first_wire]
        right = basis[(first_wire+num_wires):]
        int_basis = int(binary, 2)
        if int_basis < N:
            xmodN = format(int_basis*x % N, "0" + str(num_wires)+ "b")
            new_state.append((amplitude, left + xmodN + right))
        else:
            new_state.append(element)
    return new_state

def CxyModN(controlWire, first_wire, num_wires, x, N, input_state):
    new_state = []

    for element in input_state:
        if int(element[1][controlWire]) == 0:
            new_state.append(element)
            continue
        
        amplitude, basis = element
        binary = basis[first_wire:(first_wire+num_wires)]
        left = basis[:first_wire]
        right = basis[(first_wire+num_wires):]
        int_basis = int(binary, 2)
        if int_basis < N:
            xmodN = format(int_basis*x % N, "0" + str(num_wires) + "b")
            new_state.append((amplitude, left + xmodN + right))
        else:
            new_state.append(element)
    return new_state


def AddDuplicates(myState):
  num_bits = len(myState[0][1])
  vec = [0 for i in range(2**num_bits)]
  for state in myState:
    vec[int(state[1],2)] += np.round(state[0],4)
  new_state = []
  for i in range(len(vec)):
    elem = np.round(vec[i],4)
    if elem != 0:
        new_state.append((elem, format(i, "0" + str(num_bits) + "b")))
  return new_state

def Simulator_S(circuit):
    num_wires, myInput, myState, measure = ReadInputString(circuit)
    
    myInput = precompile(myInput)
    myState = VecToDirac(myState)
    
    circuit_len = len(myInput)
    
    i = 0
    for gate in myInput:
        name = gate[0]
        
        if name == "H":
            myState = H(int(gate[1]), myState)
        elif name == "P":
            myState = P(int(gate[1]), float(gate[2]), myState)
        elif name == "CNOT":
            myState = CNOT(int(gate[1]), int(gate[2]), myState)
        elif name == "xyModN":
            myState = xyModN(int(gate[1]), int(gate[2]), int(gate[3]), int(gate[4]), myState)
        elif name == "CxyModN":
            myState = CxyModN(int(gate[1]), int(gate[2]), int(gate[3]), int(gate[4]), int(gate[5]), myState)
        
        i += 1
        myState = AddDuplicates(myState)
    
    if measure:
        return measureState(DiracToVec(myState))
    return myState

#Simulator Ma

def HadamardArray(wire, num_wires):
    hadamard_matrix = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                                  [1/np.sqrt(2), -1/np.sqrt(2)]])

    myMatrix = 1
    for i in range(num_wires):
        if i == wire:
            myMatrix = np.kron(myMatrix, hadamard_matrix)  # Apply Hadamard on target wire
        else:
            myMatrix = np.kron(myMatrix, np.identity(2))  # Apply identity on other wires

    return myMatrix


def PhaseArray(wire, num_wires, theta):
    phase_matrix = np.array([[1, 0],
                            [0, np.exp(1j*theta)]])

    myMatrix = 1  # Start with a scalar identity
    for i in range(num_wires):
        if i == wire:
            myMatrix = np.kron(myMatrix, phase_matrix)  # Apply Hadamard on target wire
        else:
            myMatrix = np.kron(myMatrix, np.identity(2))  # Apply identity on other wires

    return myMatrix


def CNOTArray(control, output, num_wires):
    size = 2 ** num_wires

    cnot = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]]) if control < output else np.array([[1, 0, 0, 0],
                                                                       [0, 0, 0, 1],
                                                                       [0, 0, 1, 0],
                                                                       [0, 1, 0, 0]])

    myMatrix = 1
    for i in range(num_wires):
        if (control < output and i == control) or (control > output and i == control):
            newMatrix = cnot
        elif (control < output and i - 1 == control) or (control > output and i + 1 == control):
            continue
        else:
            newMatrix = np.identity(2)

        myMatrix = np.kron(myMatrix, newMatrix)

    return myMatrix

def Simulator_Ma(circuit):
    num_wires,myInput,myState,measure=ReadInputString(circuit)
    
    myInput = precompile(myInput)
    
    myMatrix = np.identity(2 ** num_wires)

    for gate in myInput:
        gate_type, *params = gate

        if gate_type == "H":
            myMatrix = HadamardArray(int(params[0]), num_wires) @ myMatrix
        elif gate_type == "P":
            myMatrix = PhaseArray(int(params[0]), num_wires, float(params[1])) @ myMatrix
        elif gate_type == "CNOT":
            myMatrix = CNOTArray(int(params[0]), int(params[1]), num_wires) @ myMatrix

    if measure:
        return measureState(myMatrix @ myState)
    return VecToDirac(myMatrix @ myState)


#Simulator Mb

def Simulator_Mb(myState):
    num_wires,myInput,myState,measure=ReadInputString(circuit)

    myInput = precompile(myInput)
    
    myMatrix = np.identity(2 ** num_wires)

    for gate in myInput:
        gate_type, *params = gate

        if gate_type == "H":
            myState = HadamardArray(int(params[0]), num_wires) @ myState
        elif gate_type == "P":
            myState = PhaseArray(int(params[0]), num_wires, float(params[1])) @ myState
        elif gate_type == "CNOT":
            myState = CNOTArray(int(params[0]), int(params[1]), num_wires) @ myState

    
    if measure:
        return measureState(myState)
          
    return VecToDirac(myState)

#Non-Atomic Gates

circuit_NOT = open("NOT_Test.txt").read()


circuit_swap = open("swap_test.txt").read()


circuit_precomp = open("precomp_test.txt").read()

#Phase Estimation

def compute_probabilities(upper_bits, total_bits, results):
    probabilities = np.zeros(2 ** upper_bits)
    
    for amplitude, bit_string in results:
        index = int(bit_string[:upper_bits], 2)
        probabilities[index] += np.abs(amplitude) ** 2
    
    return probabilities

phase_values = np.linspace(0, 2 * np.pi, 100)
estimated_phases = []

for phase_angle in phase_values:
    circuit_description = ("2 \n INITSTATE BASIS |01> \n H 0 \n "
                           f"CPHASE 0 1 {phase_angle} \n H 0")
    
    simulation_result = Simulator_S(circuit_description)
    max_prob = 0
    best_estimate = 0
    
    for amplitude, bit_string in simulation_result:
        probability = np.abs(amplitude) ** 2
        if probability > max_prob:
            max_prob = probability
            best_estimate = int(bit_string[0]) / 2
    
    estimated_phases.append(best_estimate)

plt.figure(0)
plt.plot(phase_values / (2 * np.pi), estimated_phases)
plt.title("Phase Estimation with 1 Wire")
plt.xlabel("Real Phase")
plt.ylabel("Estimated Phase")
plt.savefig("phase_est_1wire.png")

fixed_phase = 0.1432394487827058 * 2 * np.pi
circuit_description = ("2 \n INITSTATE BASIS |01> \n H 0 \n "
                       f"CPHASE 0 1 {fixed_phase} \n H 0")

simulation_result = Simulator_S(circuit_description)
probability_distribution = compute_probabilities(1, 2, simulation_result)
phase_steps = [i / len(simulation_result[0][1]) for i in range(len(simulation_result) + 1)]

plt.figure(1)
plt.stairs(probability_distribution, phase_steps, fill=True)
plt.axvline(x=0.1432394487827058, color='r', linestyle='--')
plt.title("Phase Estimation of 0.1432")
plt.xlabel("Estimated Phase")
plt.ylabel("Probability")
plt.savefig("phase_est_hist_1wire.png")
plt.show()

def invert(circuit):
    return '\n'.join(circuit.split('\n')[::-1])


def make_hadamard_section(top_wires):
    output = ""
    for i in range(top_wires):
        output += "H " + str(i) + "\n"
    return output


def make_control(control, top_wires, circuit):
    controlled_output = ""
    for line in circuit.split('\n'):
        tokens = line.split()
        if len(tokens) <= 1:
            continue
        
        gate = tokens[0]
        if gate == "NOT":
            controlled_output += "CNOT " + str(control) + " " + str(int(tokens[-1]) + top_wires) + "\n"
        elif "P" in gate:
            controlled_output += "CPHASE " + str(control) + " " + str(int(tokens[-2]) + top_wires) + " " + tokens[-1] + "\n"
        elif "xyModN" in line:
            controlled_output += "CxyModN" + " " + str(control) + " " + str(int(tokens[1]) + top_wires) + " " + tokens[2] + " " + tokens[3] + " " + tokens[4] +  "\n"
        else:
            controlled_output += line + "\n"
    
    return controlled_output


def phase_est_circuit_gen(top_wires, num_wires, phase):
    circuit = make_hadamard_section(top_wires)
    
    # Controlled phase gates

    #Slow
    # for i in range(top_wires):
    #    for j in range(2**i):
    #         circuit += "CPHASE " + str(top_wires-i-1) + " " + str(top_wires) + " " + str(phase) + "\n"
    
    #Fast
    for i in range(top_wires):
        circuit += "CPHASE " + str(top_wires - i - 1) + " " + str(top_wires) + " " + str(phase * (2 ** i)) + "\n"
    
    # Inverse Quantum Fourier Transform
    circuit += invert(QFT_Gen(top_wires))
    return circuit


def dif_phase_est_circuit(top_wires, circuit):
    output = make_hadamard_section(top_wires)
    
    # Controlled circuit section
    for i in range(top_wires):
        for _ in range(2 ** i):
            output += make_control(top_wires - i - 1, top_wires, circuit)
    
    # Inverse Quantum Fourier Transform
    output += invert(QFT_Gen(top_wires))
    return output

#Quantum Fourier Transform

def Simulator_Ma_MatGen(circuit):
    num_wires,myInput,myState,measure=ReadInputString(circuit)
    
    myInput = precompile(myInput)
    
    myMatrix = np.identity(2 ** num_wires)

    for gate in myInput:
        gate_type, *params = gate

        if gate_type == "H":
            myMatrix = HadamardArray(int(params[0]), num_wires) @ myMatrix
        elif gate_type == "P":
            myMatrix = PhaseArray(int(params[0]), num_wires, float(params[1])) @ myMatrix
        elif gate_type == "CNOT":
            myMatrix = CNOTArray(int(params[0]), int(params[1]), num_wires) @ myMatrix

    return myMatrix

circuit = open("QFT_3wires.txt").read()

matrix_QFT = Simulator_Ma_MatGen(circuit)

matrix_Test = np.array(
    [[np.round(np.power(np.sqrt(0 + 1j), (i * j)), 2) / np.sqrt(8) for j in range(8)] 
     for i in range(8)], dtype=complex
)

difference = np.round(matrix_QFT - matrix_Test, 2)

print(difference)


#QFT Circuit Generator

def QFT_Gen(num_wires):
    output = ""

    for i in range(num_wires//2):
            output += "SWAP " + str(i) + " " + str(num_wires - (i+1)) + "\n"

    
    for i in range(num_wires):
        output += "H " + str(num_wires-i-1) + "\n"
        for j in range(num_wires-i-1):
            output += "CPHASE " + str(num_wires-i-1) + " " + str(num_wires-j-i-2) + " " + str(-np.pi/(2**(j+1))) + "\n"
    return output

#Classical Shor's Algorithm

def prime_checker(num):
    if num <= 1:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    limit = math.isqrt(num)
    for i in range(3, limit + 1, 2):
        if num % i == 0:
            return False
    return True

def root_checker(num):
    max_exp = math.ceil(math.log2(num))
    for a in range(2, max_exp + 1):
        root = num ** (1 / a)
        if root == int(root):
            return True
    return False

def period_finder(x, num):
    r = 2
    while pow(x, r, num) != 1:
        r += 1
    return r

def alg_cond(num):
    if num % 2 == 0:
        return True
    if prime_checker(num):
        return True
    if root_checker(num):
        return True

def ClassicalShor(num):
    if alg_cond(num):
        return "Didn\'t meet algorithm conditions"
    
    while True:
        x = np.random.randint(2, num)
        if np.gcd(x, num) != 1:
            return(np.gcd(x, num), num//np.gcd(x, num))
        
        r = period_finder(x,num)
        #print(x, num, r)

        if r % 2 == 1:
            continue

        f_1 = np.gcd(int((x**(r//2)-1) % num),num)
        f_2 = np.gcd(int((x**(r//2)+1) % num),num)
        if f_1 != 1 and f_2 != 1 and f_1 != num and f_2 != num:
            return (f_1, f_2, x, r)

for i in range(2,2**7):
    factors = ClassicalShor(i)
    if factors != "Didn\'t meet algorithm conditions" and len(factors) == 4:
        print(factors[0],"*",factors[1], "=" , i, "(x =", factors[2], "r =", factors[3],")")
        
#Quantum Matrix

def Gen_UniMat(x, num):
    size = 2 ** math.ceil(math.log2(num))
    matrix = np.zeros((size, size))

    for i in range(size):
        row = (i * x % num) if i < num else i
        matrix[row][i] = 1

    return matrix

def period_finder_U(x, num):
    eigenvalues = np.linalg.eigvals(Gen_UniMat(x, num))
    phase_values = (np.angle(eigenvalues) + 2 * np.pi) / (2 * np.pi)

    random_phase = np.random.choice(phase_values)
    while random_phase == 0:
        random_phase = np.random.choice(phase_values)

    period = Fraction(random_phase).limit_denominator(num).denominator
    return period



def ClassicalShor_U(num):
    if alg_cond(num):
        return "Didn\'t meet algorithm conditions"
    
    while True:
        x = np.random.randint(2, num)
        if np.gcd(x, num) != 1:
            return(np.gcd(x, num), num//np.gcd(x, num))
        
        r = period_finder_U(x,num)

        if r % 2 == 1:
            continue

        f_1 = np.gcd(int((x**(r//2)-1) % num),num)
        f_2 = np.gcd(int((x**(r//2)+1) % num),num)
        if f_1 != 1 and f_2 != 1 and f_1 != num and f_2 != num:
            return (f_1, f_2,)

for i in range(2,2**8):
    factors = ClassicalShor_U(i)
    if factors != "Didn\'t meet algorithm conditions":
        print(factors[0],"*",factors[1], "=" , i)
        

#Quantum Shor's Algorithm

def shor_circuit_gen(x, num):
    num_wires_u = int(np.ceil(np.log2(num)))
    unitary_circuit = str(num_wires_u) + "\n xyModN 0 " + str(num_wires_u) + " " + str(x) + " " + str(num)
    
    total_top_wires = 2 * num_wires_u
    total_wires = total_top_wires + num_wires_u
    circuit_structure = dif_phase_est_circuit(total_top_wires, unitary_circuit)
    
    input_state = format(1, "0" + str(total_wires) + "b")
    circuit_description = str(total_wires) + "\nINITSTATE BASIS |" + input_state + "> \n" + circuit_structure
    
    return circuit_description

def quantum_period_finder(x, num):
    num_wires_u = int(np.ceil(np.log2(num)))
    total_top_wires = 2 * num_wires_u
    
    output_data = Simulator_S(shor_circuit_gen(x, num) + "\nMEASURE")[1:total_top_wires+1]
    print(output_data)
    
    max_amplitude = 0
    best_estimate = 0

    best_estimate = int(output_data, 2) / (2 ** total_top_wires)
    period = Fraction(best_estimate).limit_denominator(num).denominator
    return period

def QuantumShor(num):
    if alg_cond(num):
        return "Didn\'t meet algorithm conditions"
    
    while True:
        x = np.random.randint(2, num)
        if np.gcd(x, num) != 1:
            return(np.gcd(x, num), num//np.gcd(x, num))
        
        r = quantum_period_finder(x,num)
        #print(x, num, r)

        if r % 2 == 1:
            continue

        f_1 = np.gcd(int((x**(r//2)-1) % num),num)
        f_2 = np.gcd(int((x**(r//2)+1) % num),num)
        if f_1 != 1 and f_2 != 1 and f_1 != num and f_2 != num:
            return (f_1, f_2)


def fast_shor_circuit_gen(x, num):
    num_wires_u = int(np.ceil(np.log2(num)))
    total_top_wires = 2 * num_wires_u
    total_wires = total_top_wires + num_wires_u

    output = make_hadamard_section(total_top_wires)
  
    new_x = x
    for i in range(total_top_wires):
        output += "CxyModN " + str(total_top_wires - 1 - i) + " " + str(total_top_wires) + " " + str(num_wires_u) + " " + str(new_x) + " " + str(num) + "\n"
        new_x = (new_x*new_x) % num
    
    # Inverse Quantum Fourier Transform
    output += invert(QFT_Gen(total_top_wires))

    
    input_state = format(1, "0" + str(total_wires) + "b")
    circuit_description = str(total_wires) + "\nINITSTATE BASIS |" + input_state + "> \n" + output

    return circuit_description

def fast_quantum_period_finder(x, num):
    num_wires_u = int(np.ceil(np.log2(num)))
    total_top_wires = 2 * num_wires_u
    
    output_data = Simulator_S(fast_shor_circuit_gen(x, num) + "\nMEASURE")[1:total_top_wires+1]
    
    max_amplitude = 0
    best_estimate = 0


    best_estimate = int(output_data, 2) / (2 ** total_top_wires)
    print(best_estimate)
    period = Fraction(best_estimate).limit_denominator(num).denominator
    return period

def fast_QuantumShor(num):
    if alg_cond(num):
        return "Didn\'t meet algorithm conditions"
    
    while True:
        x = np.random.randint(2, num)
        if np.gcd(x, num) != 1:
            return(np.gcd(x, num), num//np.gcd(x, num))
        
        r = fast_quantum_period_finder(x,num)

        if r % 2 == 1:
            continue

        f_1 = np.gcd(int((x**(r//2)-1) % num),num)
        f_2 = np.gcd(int((x**(r//2)+1) % num),num)
        if f_1 != 1 and f_2 != 1 and f_1 != num and f_2 != num:
            return (f_1, f_2)


print(fast_QuantumShor(15))
