import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  VStack,
  Text,
  Icon,
  Progress,
  HStack,
  Badge,
} from '@chakra-ui/react';
import { FiUpload, FiFile, FiCheckCircle } from 'react-icons/fi';

const FileUpload = ({ onFileUpload }) => {
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();

      reader.onprogress = (event) => {
        if (event.lengthComputable) {
          const progress = (event.loaded / event.total) * 100;
          setUploadProgress(progress);
        }
      };

      reader.onload = () => {
        setUploadProgress(100);
        if (file.type === 'application/pdf') {
          onFileUpload({
            name: file.name,
            content: reader.result,
            type: file.type,
            isPDF: true
          });
        } else {
          onFileUpload({
            name: file.name,
            content: reader.result,
            type: file.type
          });
        }
        setTimeout(() => setUploadProgress(0), 1000);
      };

      if (file.type === 'application/pdf') {
        reader.readAsArrayBuffer(file);
      } else {
        reader.readAsText(file);
      }
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'chemical/x-mol': ['.mol'],
      'chemical/x-pdb': ['.pdb'],
      'text/plain': ['.txt', '.smi']
    },
    multiple: false
  });

  return (
    <Box
      {...getRootProps()}
      border="2px dashed"
      borderColor={isDragActive ? 'blue.400' : 'whiteAlpha.300'}
      borderRadius="2xl"
      p={8}
      cursor="pointer"
      transition="all 0.3s"
      _hover={{
        borderColor: 'blue.400',
        bg: 'whiteAlpha.100',
        transform: 'translateY(-2px)',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
      }}
      position="relative"
      bg={isDragActive ? 'rgba(66, 153, 225, 0.08)' : 'whiteAlpha.50'}
    >
      <input {...getInputProps()} />
      <VStack spacing={4}>
        <Icon
          as={isDragActive ? FiFile : FiUpload}
          boxSize={10}
          color={isDragActive ? 'blue.400' : 'blue.300'}
          transition="all 0.3s"
        />
        <VStack spacing={2}>
          <Text
            textAlign="center"
            fontSize="lg"
            fontWeight="medium"
            color="white"
            transition="all 0.3s"
          >
            {isDragActive ? 'Drop to upload' : 'Drag & drop or click to upload'}
          </Text>
          <HStack spacing={3} flexWrap="wrap" justify="center">
            <Badge
              colorScheme="blue"
              variant="subtle"
              px={3}
              py={1}
              borderRadius="full"
              fontSize="sm"
            >
              PDF
            </Badge>
            <Badge
              colorScheme="green"
              variant="subtle"
              px={3}
              py={1}
              borderRadius="full"
              fontSize="sm"
            >
              MOL
            </Badge>
            <Badge
              colorScheme="purple"
              variant="subtle"
              px={3}
              py={1}
              borderRadius="full"
              fontSize="sm"
            >
              PDB
            </Badge>
            <Badge
              colorScheme="orange"
              variant="subtle"
              px={3}
              py={1}
              borderRadius="full"
              fontSize="sm"
            >
              SMILES
            </Badge>
          </HStack>
        </VStack>
      </VStack>

      {uploadProgress > 0 && (
        <Box
          position="absolute"
          bottom={0}
          left={0}
          right={0}
          px={8}
          pb={6}
          bg={uploadProgress === 100 ? 'rgba(66, 153, 225, 0.08)' : 'transparent'}
          borderBottomRadius="2xl"
          transition="all 0.3s"
        >
          <Progress
            value={uploadProgress}
            size="sm"
            colorScheme="blue"
            borderRadius="full"
            bg="whiteAlpha.200"
            hasStripe={uploadProgress < 100}
            isAnimated={uploadProgress < 100}
          />
          {uploadProgress === 100 && (
            <HStack spacing={2} justify="center" mt={3}>
              <Icon as={FiCheckCircle} color="green.400" boxSize={5} />
              <Text color="green.400" fontSize="md" fontWeight="medium">
                Upload complete
              </Text>
            </HStack>
          )}
        </Box>
      )}
    </Box>
  );
};

export default FileUpload;
