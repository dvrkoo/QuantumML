import pennylane as qml


def local_pauli_group(qubits: int, locality: int):
    assert locality <= qubits, f"Locality must not exceed the number of qubits."
    return list(generate_paulis(0, 0, "", qubits, locality))


# This is a recursive generator function that constructs Pauli strings.
def generate_paulis(
    identities: int, paulis: int, output: str, qubits: int, locality: int
):
    # Base case: if the output string's length matches the number of qubits, yield it.
    if len(output) == qubits:
        yield output
    else:
        # Recursive case: add an "I" (identity) to the output string.
        yield from generate_paulis(
            identities + 1, paulis, output + "I", qubits, locality
        )

        # If the number of Pauli operators used is less than the locality, add "X", "Y", or "Z"
        # systematically builds all possible Pauli strings that conform to the specified locality.
        if paulis < locality:
            yield from generate_paulis(
                identities, paulis + 1, output + "X", qubits, locality
            )
            yield from generate_paulis(
                identities, paulis + 1, output + "Y", qubits, locality
            )
            yield from generate_paulis(
                identities, paulis + 1, output + "Z", qubits, locality
            )
