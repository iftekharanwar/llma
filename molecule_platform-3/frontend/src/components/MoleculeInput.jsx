import React, { useState } from 'react';
import {
  Box,
  FormControl,
  FormLabel,
  Textarea,
  Select,
  Button,
  VStack,
  useToast,
  Text,
  Alert,
  AlertIcon,
  AlertDescription,
  Divider,
  Icon,
  HStack,
} from '@chakra-ui/react';
import { FiUpload, FiFileText, FiAlertCircle } from 'react-icons/fi';
import FileUpload from './FileUpload';

const FORMATS = {
  smiles: {
    name: 'SMILES',
    description: 'Simplified Molecular Input Line Entry System - e.g., CC(=O)OC1=CC=CC=C1C(=O)O for Aspirin',
    placeholder: 'Enter SMILES notation...',
  },
  mol: {
    name: 'MOL',
    description: '2D/3D molecular structure format',
    placeholder: 'Paste MOL file content...',
  },
  pdb: {
    name: 'PDB',
    description: 'Protein Data Bank format for 3D structures',
    placeholder: 'Paste PDB file content...',
  },
};

const CONTEXTS = {
  medicine: 'Pharmaceutical/Medical Use',
  pesticide: 'Agricultural/Pesticide Use',
  sports: 'Sports Supplement',
  food: 'Food/Beverage Additive',
};

const MoleculeInput = ({ onAnalyze }) => {
  const [structure, setStructure] = useState('');
  const [inputFormat, setInputFormat] = useState('smiles');
  const [context, setContext] = useState('medicine');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const toast = useToast();

  const validateInput = () => {
    if (!structure.trim()) {
      setError('Please enter a molecular structure');
      return false;
    }

    // Basic format-specific validation
    if (inputFormat === 'smiles' && !/^[A-Za-z0-9()=#@+\-\[\]\\\/\s]+$/.test(structure)) {
      setError('Invalid SMILES format. Please check your input.');
      return false;
    }

    if (inputFormat === 'mol' && !structure.includes('M  END')) {
      setError('Invalid MOL format. Must contain "M  END" marker.');
      return false;
    }

    if (inputFormat === 'pdb' && !structure.includes('ATOM') && !structure.includes('HETATM')) {
      setError('Invalid PDB format. Must contain ATOM or HETATM records.');
      return false;
    }

    setError('');
    return true;
  };

  const handleFileUpload = async (fileData) => {
    if (fileData.type === 'application/pdf') {
      setIsLoading(true);
      try {
        const formData = new FormData();
        const pdfBlob = new Blob([fileData.content], { type: 'application/pdf' });
        formData.append('file', pdfBlob, fileData.name);

        const response = await fetch('http://localhost:8005/api/process-pdf', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'PDF processing failed');
        }

        // If we found potential SMILES in the PDF, use the first one
        if (data.potential_structures.smiles.length > 0) {
          const potentialSmiles = data.potential_structures.smiles.find(s =>
            /^[A-Za-z0-9()=#@+\-\[\]\\\/\s]+$/.test(s)
          );
          if (potentialSmiles) {
            setStructure(potentialSmiles);
            setInputFormat('smiles');
            toast({
              title: 'PDF Processed',
              description: 'Found and extracted molecular structure',
              status: 'success',
              duration: 3000,
            });
            return;
          }
        }

        toast({
          title: 'PDF Processed',
          description: 'No valid molecular structures found in PDF',
          status: 'warning',
          duration: 3000,
        });
      } catch (error) {
        toast({
          title: 'Error',
          description: error.message,
          status: 'error',
          duration: 5000,
        });
      } finally {
        setIsLoading(false);
      }
      return;
    }

    setStructure(fileData.content);
    // Detect format from file extension
    const ext = fileData.name.split('.').pop().toLowerCase();
    const formatMap = {
      'mol': 'mol',
      'pdb': 'pdb',
      'smi': 'smiles',
      'txt': 'smiles', // Assume SMILES for .txt files
    };
    if (formatMap[ext]) {
      setInputFormat(formatMap[ext]);
    }
  };

  const handleSubmit = async () => {
    if (!validateInput()) return;

    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          structure,
          input_format: inputFormat,
          context,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Analysis failed');
      }

      onAnalyze(data);
      toast({
        title: 'Analysis Complete',
        description: 'Molecular structure analyzed successfully',
        status: 'success',
        duration: 3000,
      });
    } catch (error) {
      setError(error.message);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 5000,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <VStack spacing={6} align="stretch">
      <Box
        bg="rgba(255, 255, 255, 0.03)"
        p={8}
        borderRadius="2xl"
        border="1px solid"
        borderColor="whiteAlpha.100"
        _hover={{ borderColor: 'blue.400', boxShadow: '0 8px 16px rgba(0, 0, 0, 0.2)' }}
        transition="all 0.3s"
      >
        <HStack spacing={3} mb={6}>
          <Icon as={FiUpload} color="blue.400" boxSize={6} />
          <Text fontSize="xl" fontWeight="semibold" color="white">
            Upload Molecule Data
          </Text>
        </HStack>
        <FileUpload onFileUpload={handleFileUpload} />
      </Box>

      <Box
        bg="rgba(255, 255, 255, 0.03)"
        p={8}
        borderRadius="2xl"
        border="1px solid"
        borderColor="whiteAlpha.100"
        _hover={{ borderColor: 'blue.400', boxShadow: '0 8px 16px rgba(0, 0, 0, 0.2)' }}
        transition="all 0.3s"
      >
        <HStack spacing={3} mb={6}>
          <Icon as={FiFileText} color="blue.400" boxSize={6} />
          <Text fontSize="xl" fontWeight="semibold" color="white">
            Manual Input
          </Text>
        </HStack>

        <VStack spacing={6}>
          <FormControl>
            <FormLabel color="blue.200" fontSize="md" fontWeight="medium">Input Format</FormLabel>
            <Select
              value={inputFormat}
              onChange={(e) => {
                setInputFormat(e.target.value);
                setStructure('');
                setError('');
              }}
              bg="whiteAlpha.50"
              borderColor="whiteAlpha.200"
              _hover={{ borderColor: 'blue.400' }}
              _focus={{ borderColor: 'blue.400', boxShadow: '0 0 0 1px var(--chakra-colors-blue-400)' }}
              color="white"
              size="lg"
            >
              {Object.entries(FORMATS).map(([value, { name }]) => (
                <option key={value} value={value} style={{color: 'black'}}>{name}</option>
              ))}
            </Select>
            <Text fontSize="sm" color="whiteAlpha.700" mt={2}>
              {FORMATS[inputFormat].description}
            </Text>
          </FormControl>

          <FormControl>
            <FormLabel color="blue.200" fontSize="md" fontWeight="medium">Analysis Context</FormLabel>
            <Select
              value={context}
              onChange={(e) => setContext(e.target.value)}
              bg="whiteAlpha.50"
              borderColor="whiteAlpha.200"
              _hover={{ borderColor: 'blue.400' }}
              _focus={{ borderColor: 'blue.400', boxShadow: '0 0 0 1px var(--chakra-colors-blue-400)' }}
              color="white"
              size="lg"
            >
              {Object.entries(CONTEXTS).map(([value, label]) => (
                <option key={value} value={value} style={{color: 'black'}}>{label}</option>
              ))}
            </Select>
          </FormControl>

          <FormControl isInvalid={!!error}>
            <FormLabel color="blue.200" fontSize="md" fontWeight="medium">Structure Input</FormLabel>
            <Textarea
              value={structure}
              onChange={(e) => {
                setStructure(e.target.value);
                setError('');
              }}
              placeholder={FORMATS[inputFormat].placeholder}
              rows={6}
              fontFamily="mono"
              bg="whiteAlpha.50"
              borderColor="whiteAlpha.200"
              _hover={{ borderColor: 'blue.400' }}
              _focus={{ borderColor: 'blue.400', boxShadow: '0 0 0 1px var(--chakra-colors-blue-400)' }}
              color="white"
              _placeholder={{ color: 'whiteAlpha.500' }}
              size="lg"
              fontSize="md"
            />
            {error && (
              <Alert status="error" mt={3} bg="red.900" color="white" borderRadius="lg">
                <AlertIcon as={FiAlertCircle} color="red.200" />
                <AlertDescription fontSize="md">{error}</AlertDescription>
              </Alert>
            )}
          </FormControl>
        </VStack>
      </Box>

      <Button
        colorScheme="blue"
        onClick={handleSubmit}
        isLoading={isLoading}
        loadingText="Analyzing..."
        size="lg"
        height="60px"
        fontSize="lg"
        fontWeight="semibold"
        w="full"
        mt={4}
        _hover={{
          transform: 'translateY(-2px)',
          boxShadow: '0 8px 16px rgba(0, 0, 0, 0.2)',
        }}
        transition="all 0.2s"
      >
        Analyze Structure
      </Button>
    </VStack>
  );
};

export default MoleculeInput;
