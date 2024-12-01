import React, { useState, useEffect } from 'react';
import { Box, Grid, Paper, Typography, CircularProgress } from '@mui/material';
import SimilaritySearchResults from './SimilaritySearchResults';
import RiskAssessment from './RiskAssessment';
import MedicalRecordsDashboard from './MedicalRecordsDashboard';
import MoleculeViewer3D from './MoleculeViewer3D';

const ComprehensiveReport = ({ moleculeData, context }) => {
  const [loading, setLoading] = useState(true);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            structure: moleculeData.structure,
            format: moleculeData.format,
            context: context
          }),
        });

        if (!response.ok) {
          throw new Error(`Analysis failed: ${response.statusText}`);
        }

        const data = await response.json();
        setReport(data);
        setError(null);
      } catch (err) {
        setError(err.message);
        console.error('Error fetching report:', err);
      } finally {
        setLoading(false);
      }
    };

    if (moleculeData?.structure) {
      fetchReport();
    }
  }, [moleculeData, context]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Typography color="error" variant="h6">
          Error: {error}
        </Typography>
      </Box>
    );
  }

  if (!report) {
    return null;
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Grid container spacing={3}>
        {/* 3D Visualization */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '400px' }}>
            <Typography variant="h6" gutterBottom>
              Molecular Structure
            </Typography>
            <MoleculeViewer3D
              structure={moleculeData.structure}
              format={moleculeData.format}
            />
          </Paper>
        </Grid>

        {/* Similarity Search Results */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '400px', overflow: 'auto' }}>
            <SimilaritySearchResults
              results={report.similarity_search_results}
              onMoleculeSelect={(molecule) => {
                // Handle molecule selection
              }}
            />
          </Paper>
        </Grid>

        {/* Risk Assessment */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <RiskAssessment
              assessment={report.risk_assessment}
              context={context}
            />
          </Paper>
        </Grid>

        {/* Medical Records Dashboard */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <MedicalRecordsDashboard
              medicalData={report.medical_records_analysis}
              moleculeName={moleculeData.name}
            />
          </Paper>
        </Grid>

        {/* Analysis Metadata */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Analysis Quality Metrics
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="subtitle1">
                  Data Completeness: {report.analysis_metadata.data_completeness}%
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="subtitle1">
                  Confidence Score: {report.analysis_metadata.confidence_score}%
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ComprehensiveReport;
